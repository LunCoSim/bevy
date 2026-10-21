use super::ShaderDefVal;
use alloc::borrow::Cow;
use bevy_asset::{
    io::AssetSourceId, io::Reader, Asset, AssetLoader, AssetPath, AssetServer, AssetServerMode,
    Handle, LoadContext,
};
use bevy_reflect::TypePath;
use bevy_utils::define_atomic_id;
use thiserror::Error;

fn scan_wesl_imports(
    source: &str,
    self_module_path: &wesl::syntax::ModulePath,
) -> Vec<ShaderImport> {
    use wesl::syntax::{ImportContent, ModulePath, PathOrigin};

    fn leaves(content: &ImportContent, path: ModulePath, out: &mut Vec<ModulePath>) {
        match content {
            ImportContent::Item(item) => {
                let mut full = path.clone();
                full.push(&item.ident.to_string());
                out.push(path);
                out.push(full);
            }
            ImportContent::Collection(collection) => {
                for import in collection {
                    let path = path.clone().join(import.path.iter().cloned());
                    leaves(&import.content, path, out);
                }
            }
        }
    }

    let Ok(translation_unit) = source.parse::<wesl::syntax::TranslationUnit>() else {
        return Vec::new();
    };

    let mut paths = Vec::new();
    for statement in &translation_unit.imports {
        match &statement.path {
            Some(import_path) => {
                let path = self_module_path.join_path(import_path);
                leaves(&statement.content, path, &mut paths);
            }
            None => {
                if let ImportContent::Collection(collection) = &statement.content {
                    for import in collection {
                        let mut components = import.path.iter().cloned();
                        if let Some(package) = components.next() {
                            let path =
                                ModulePath::new(PathOrigin::Package(package), components.collect());
                            leaves(&import.content, path, &mut paths);
                        }
                    }
                }
            }
        }
    }

    let mut imports = Vec::new();
    for path in &paths {
        let path = match &path.origin {
            PathOrigin::Package(pkg) if pkg.contains('/') => Cow::Owned(ModulePath {
                origin: PathOrigin::Package(pkg.rsplit('/').next().unwrap().to_string()),
                components: path.components.clone(),
            }),
            _ => Cow::Borrowed(path),
        };
        let import = match &path.origin {
            PathOrigin::Absolute => {
                ShaderImport::AssetPath(format!("/{}", path.components.join("/")))
            }
            PathOrigin::Package(package) => ShaderImport::Custom(
                core::iter::once(package.as_str())
                    .chain(path.components.iter().map(String::as_str))
                    .collect::<Vec<_>>()
                    .join("::"),
            ),
            PathOrigin::Relative(_) => continue,
        };
        if !imports.contains(&import) {
            imports.push(import);
        }
    }
    imports
}

define_atomic_id!(ShaderId);

/// Describes whether or not to perform runtime checks on shaders.
/// Runtime checks can be enabled for safety at the cost of speed.
/// By default no runtime checks will be performed.
///
/// # Panics
/// Because no runtime checks are performed for spirv,
/// enabling `ValidateShader` for spirv will cause a panic
#[derive(Clone, Debug, Default)]
pub enum ValidateShader {
    #[default]
    /// No runtime checks for soundness (e.g. bound checking) are performed.
    ///
    /// This is suitable for trusted shaders, written by your program or dependencies you trust.
    Disabled,
    /// Enable's runtime checks for soundness (e.g. bound checking).
    ///
    /// While this can have a meaningful impact on performance,
    /// this setting should *always* be enabled when loading untrusted shaders.
    /// This might occur if you are creating a shader playground, running user-generated shaders
    /// (as in `VRChat`), or writing a web browser in Bevy.
    Enabled,
}

/// An "unprocessed" shader. It can contain imports and conditional
/// compilation attributes.
#[derive(Asset, TypePath, Debug, Clone)]
pub struct Shader {
    /// The asset path of the shader.
    pub path: String,
    /// The raw source code of the shader.
    pub source: Source,
    /// The path from which this shader can be imported by other shaders.
    pub import_path: ShaderImport,
    /// The import paths this shader depends on.
    pub imports: Vec<ShaderImport>,
    /// Any shader defs that should be included when this module is used.
    pub shader_defs: Vec<ShaderDefVal>,
    /// Strong handles to this shader's dependencies, to prevent them
    /// from being immediately dropped if this shader is the only user.
    pub file_dependencies: Vec<Handle<Shader>>,
    /// Enable or disable runtime shader validation, trading safety against speed.
    ///
    /// Please read the [`ValidateShader`] docs for a discussion of the tradeoffs involved.
    pub validate_shader: ValidateShader,
}

impl Shader {
    /// Creates a new WGSL shader.
    pub fn from_wgsl(source: impl Into<Cow<'static, str>>, path: impl Into<String>) -> Shader {
        let source = source.into();
        let path = path.into();
        Shader {
            import_path: ShaderImport::AssetPath(path.clone()),
            path,
            imports: Vec::new(),
            source: Source::Wgsl(source),
            shader_defs: Default::default(),
            file_dependencies: Default::default(),
            validate_shader: ValidateShader::Disabled,
        }
    }

    /// Creates a new WGSL shader with some given shader defs.
    pub fn from_wgsl_with_defs(
        source: impl Into<Cow<'static, str>>,
        path: impl Into<String>,
        shader_defs: Vec<ShaderDefVal>,
    ) -> Shader {
        Self {
            shader_defs,
            ..Self::from_wgsl(source, path)
        }
    }

    /// Creates a new SPIR-V shader.
    pub fn from_spirv(source: impl Into<Cow<'static, [u8]>>, path: impl Into<String>) -> Shader {
        let path = path.into();
        Shader {
            path: path.clone(),
            imports: Vec::new(),
            import_path: ShaderImport::AssetPath(path),
            source: Source::SpirV(source.into()),
            shader_defs: Default::default(),
            file_dependencies: Default::default(),
            validate_shader: ValidateShader::Disabled,
        }
    }

    /// Creates an edited WESL shader and retains imported modules through the
    /// server's registered asset readers. Run this future off the main thread.
    /// Unlike `from_wesl`, this prepares file dependencies for an in-memory edit.
    pub async fn from_wesl_with_dependencies(
        source: impl Into<Cow<'static, str>>,
        path: impl Into<String>,
        asset_server: &AssetServer,
    ) -> Shader {
        let mut shader = Self::from_wesl(source, path);
        for candidate in shader.wesl_dependency_candidates() {
            let path = AssetPath::from(candidate);
            // A package can be supplied by embedded shaders rather than a
            // registered asset source. Only file-backed modules have handles.
            let Ok(source) = asset_server.get_source(path.source()) else {
                continue;
            };
            let reader = match asset_server.mode() {
                AssetServerMode::Unprocessed => source.reader(),
                AssetServerMode::Processed => match source.processed_reader() {
                    Ok(reader) => reader,
                    Err(_) => continue,
                },
            };
            if reader.read(path.path()).await.is_ok() {
                shader.file_dependencies.push(asset_server.load(path));
            }
        }
        shader
    }

    fn wesl_dependency_candidates(&self) -> Vec<String> {
        self.imports
            .iter()
            .filter_map(|import| match import {
                ShaderImport::AssetPath(path) => {
                    Some(format!("{}.wesl", path.trim_start_matches('/')))
                }
                ShaderImport::Custom(module) => {
                    let (source, path) = module.split_once("::")?;
                    Some(format!("{source}://{}.wesl", path.replace("::", "/")))
                }
            })
            .collect()
    }

    /// Creates a new Wesl shader.
    pub fn from_wesl(source: impl Into<Cow<'static, str>>, path: impl Into<String>) -> Shader {
        let source = source.into();
        let path = path.into();

        let import_path = match path.strip_prefix("embedded://") {
            Some(embedded_path) => ShaderImport::Custom(
                std::path::Path::new(embedded_path)
                    .with_extension("")
                    .to_string_lossy()
                    .split('/')
                    .filter(|component| !component.is_empty())
                    .collect::<Vec<_>>()
                    .join("::"),
            ),
            None => {
                let asset_path = AssetPath::parse(path.as_str());
                let stem = asset_path.path().with_extension("");
                let stem = stem.to_string_lossy().replace('\\', "/");
                match asset_path.source() {
                    AssetSourceId::Default => {
                        ShaderImport::AssetPath(format!("/{}", stem.trim_start_matches('/')))
                    }
                    AssetSourceId::Name(source) => ShaderImport::Custom(
                        core::iter::once(source.as_ref())
                            .chain(stem.split('/').filter(|part| !part.is_empty()))
                            .collect::<Vec<_>>()
                            .join("::"),
                    ),
                }
            }
        };

        let imports = crate::shader_cache::wesl_module_path(&import_path)
            .map(|module_path| scan_wesl_imports(&source, &module_path))
            .unwrap_or_default();

        Shader {
            path,
            imports,
            import_path,
            source: Source::Wesl(source),
            shader_defs: Default::default(),
            file_dependencies: Default::default(),
            validate_shader: ValidateShader::Disabled,
        }
    }
}

/// Raw shader source code.
#[expect(missing_docs, reason = "The variants are self-explanatory.")]
#[derive(Debug, Clone)]
pub enum Source {
    Wgsl(Cow<'static, str>),
    Wesl(Cow<'static, str>),
    SpirV(Cow<'static, [u8]>),
    // TODO: consider the following
    // PrecompiledSpirVMacros(HashMap<HashSet<String>, Vec<u32>>)
    // NagaModule(Module) ... Module impls Serialize/Deserialize
}

impl Source {
    /// The underlying source code string, unless it is SPIR-V.
    pub fn as_str(&self) -> &str {
        match self {
            Source::Wgsl(s) | Source::Wesl(s) => s,
            Source::SpirV(_) => panic!("spirv not yet implemented"),
        }
    }
}

/// The [`AssetLoader`] responsible for loading unprocessed shader assets.
#[derive(Default, TypePath)]
pub struct ShaderLoader;

/// An error encountered while loading a shader's source.
#[non_exhaustive]
#[derive(Debug, Error)]
#[expect(missing_docs, reason = "The variants are self-explanatory.")]
pub enum ShaderLoaderError {
    #[error("Could not load shader: {0}")]
    Io(#[from] std::io::Error),
    #[error("Could not parse shader: {0}")]
    Parse(#[from] alloc::string::FromUtf8Error),
}

/// Settings for loading shaders.
#[derive(serde::Serialize, serde::Deserialize, Debug, Default)]
pub struct ShaderSettings {
    /// The shader defs to apply when this shader is loaded.
    pub shader_defs: Vec<ShaderDefVal>,
}

impl AssetLoader for ShaderLoader {
    type Asset = Shader;
    type Settings = ShaderSettings;
    type Error = ShaderLoaderError;
    async fn load(
        &self,
        reader: &mut dyn Reader,
        settings: &Self::Settings,
        load_context: &mut LoadContext<'_>,
    ) -> Result<Shader, Self::Error> {
        let ext = load_context
            .path()
            .path()
            .extension()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let path = load_context.path().to_string();
        // On windows, the path will inconsistently use \ or /.
        // TODO: remove this once AssetPath forces cross-platform "slash" consistency. See #10511
        let path = path.replace(std::path::MAIN_SEPARATOR, "/");
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        if ext != "wesl" && !settings.shader_defs.is_empty() {
            tracing::warn!(
                "Tried to load a non-wesl shader with shader defs, this isn't supported: \
                    The shader defs will be ignored."
            );
        }
        let mut shader = match ext.as_str() {
            "spv" => Shader::from_spirv(bytes, load_context.path().path().to_string_lossy()),
            "wgsl" => Shader::from_wgsl(String::from_utf8(bytes)?, path),
            "wesl" => {
                let mut shader = Shader::from_wesl(String::from_utf8(bytes)?, path);
                shader.shader_defs = settings.shader_defs.clone();
                shader
            }
            _ => panic!("unhandled extension: {ext}"),
        };

        // collect and store file dependencies
        match ext.as_str() {
            "wesl" => {
                let candidates = shader.wesl_dependency_candidates();
                for file_path in candidates {
                    if load_context
                        .read_asset_bytes(AssetPath::from(file_path.clone()))
                        .await
                        .is_ok()
                    {
                        shader
                            .file_dependencies
                            .push(load_context.load(AssetPath::from(file_path)));
                    }
                }
            }
            _ => {
                for import in &shader.imports {
                    if let ShaderImport::AssetPath(asset_path) = import {
                        shader.file_dependencies.push(load_context.load(asset_path));
                    }
                }
            }
        }
        Ok(shader)
    }

    fn extensions(&self) -> &[&str] {
        &["spv", "wgsl", "wesl"]
    }
}

/// A shader import, described as either an asset path or an import path.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum ShaderImport {
    /// An asset path to a shader.
    AssetPath(String),
    /// An import path from which a shader may be imported.
    Custom(String),
}

/// A reference to a shader asset.
#[derive(Default)]
pub enum ShaderRef {
    /// Use the "default" shader for the current context.
    #[default]
    Default,
    /// A handle to a shader stored in the [`Assets<Shader>`](bevy_asset::Assets) resource.
    Handle(Handle<Shader>),
    /// An asset path leading to a shader.
    Path(AssetPath<'static>),
}

impl From<Handle<Shader>> for ShaderRef {
    fn from(handle: Handle<Shader>) -> Self {
        Self::Handle(handle)
    }
}

impl From<AssetPath<'static>> for ShaderRef {
    fn from(path: AssetPath<'static>) -> Self {
        Self::Path(path)
    }
}

impl From<&'static str> for ShaderRef {
    fn from(path: &'static str) -> Self {
        Self::Path(AssetPath::from(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_app::{App, TaskPoolPlugin};
    use bevy_asset::{
        io::{
            memory::{Dir, MemoryAssetReader},
            AssetSourceBuilder,
        },
        AssetApp, AssetMetaCheck, AssetPlugin, AssetServer, Assets,
    };
    use std::{
        path::Path,
        time::{Duration, Instant},
    };

    fn shader_app(directory: Dir) -> App {
        let mut app = App::new();
        let library = directory.clone();
        app.register_asset_source(
            "library",
            AssetSourceBuilder::new(move || {
                Box::new(MemoryAssetReader {
                    root: library.clone(),
                })
            }),
        );
        app.register_asset_source(
            "scoped",
            AssetSourceBuilder::new(move || {
                Box::new(MemoryAssetReader {
                    root: directory.clone(),
                })
            }),
        );
        app.add_plugins((
            TaskPoolPlugin::default(),
            AssetPlugin {
                meta_check: AssetMetaCheck::Never,
                ..Default::default()
            },
        ));
        app.init_asset::<Shader>()
            .init_asset_loader::<ShaderLoader>();
        app.finish();
        app.cleanup();
        app
    }

    fn until(app: &mut App, ready: impl Fn(&App) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            app.update();
            if ready(app) {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("shader asset did not reach its expected state");
    }

    #[test]
    fn named_source_imports_have_the_same_identity_as_their_dependency() {
        for source in ["lunco", "twin"] {
            let prefix = format!("{source}://mount/shaders");
            let root = Shader::from_wesl(
                "import super::nested::helper::value;",
                format!("{prefix}/root.wesl"),
            );
            let dependency = Shader::from_wesl(
                "fn value() -> f32 { return 1.0; }",
                format!("{prefix}/nested/helper.wesl"),
            );
            assert!(root.imports.contains(&dependency.import_path));
            assert_eq!(
                dependency.import_path,
                ShaderImport::Custom(format!("{source}::mount::shaders::nested::helper"))
            );
        }
    }

    #[test]
    fn edited_source_retains_new_imports_without_preloading() {
        let directory = Dir::default();
        directory.insert_asset(
            Path::new("shared.wesl"),
            b"fn value() -> f32 { return 0.75; }".as_slice(),
        );
        let mut app = shader_app(directory);
        let server = app.world().resource::<AssetServer>().clone();
        let shader = bevy_tasks::block_on(Shader::from_wesl_with_dependencies(
            "import library::shared::value; @fragment fn fragment() -> @location(0) vec4<f32> { return vec4(value()); }",
            "scoped://edited.wesl", &server,
        ));
        assert_eq!(shader.file_dependencies.len(), 1);
        let dependency = shader.file_dependencies[0].clone();
        until(&mut app, |_| {
            server.is_loaded_with_dependencies(dependency.id())
        });
        let mut cache = crate::ShaderCache::new((), |_, source, _| match source {
            crate::ShaderCacheSource::Wgsl(source) => Ok(source),
            _ => unreachable!(),
        });
        for (id, shader) in app.world().resource::<Assets<Shader>>().iter() {
            cache.set_shader(id, shader.clone());
        }
        let id = bevy_asset::AssetId::default();
        cache.set_shader(id, shader);
        assert!(cache.get(0, id, &[]).unwrap().contains("0.75"));
    }

    #[test]
    fn windows_drive_is_not_an_asset_source_namespace() {
        let root = Shader::from_wesl("import super::helper::value;", "C:/shaders/root.wesl");
        let helper = Shader::from_wesl("fn value() {}", "C:/shaders/helper.wesl");
        assert!(root.imports.contains(&helper.import_path));
    }

    #[test]
    fn loader_retains_nested_dependencies_from_the_named_source_without_a_gpu() {
        let directory = Dir::default();
        directory.insert_asset(Path::new("shaders/root.wesl"), b"import super::nested::helper::value; @fragment fn fragment() -> @location(0) vec4<f32> { return vec4(value()); }".as_slice());
        directory.insert_asset(
            Path::new("shaders/nested/helper.wesl"),
            b"import super::super::value::number; fn value() -> f32 { return number(); }"
                .as_slice(),
        );
        directory.insert_asset(
            Path::new("shaders/value.wesl"),
            b"fn number() -> f32 { return 0.25; }".as_slice(),
        );
        let mut app = shader_app(directory);
        let handle = app
            .world()
            .resource::<AssetServer>()
            .load::<Shader>("scoped://shaders/root.wesl");
        until(&mut app, |app| {
            app.world()
                .resource::<AssetServer>()
                .is_loaded_with_dependencies(handle.id())
        });
        let shaders = app.world().resource::<Assets<Shader>>();
        let root = shaders.get(&handle).unwrap();
        assert_eq!(root.file_dependencies.len(), 1);
        let helper = shaders.get(&root.file_dependencies[0]).unwrap();
        assert_eq!(helper.path, "scoped://shaders/nested/helper.wesl");
        assert_eq!(helper.file_dependencies.len(), 1);
        assert_eq!(
            shaders.get(&helper.file_dependencies[0]).unwrap().path,
            "scoped://shaders/value.wesl"
        );
        let mut cache = crate::ShaderCache::new((), |_, source, _| match source {
            crate::ShaderCacheSource::Wgsl(source) => Ok(source),
            _ => unreachable!(),
        });
        for (id, shader) in shaders.iter() {
            cache.set_shader(id, shader.clone());
        }
        let compiled = cache.get(0, handle.id(), &[]).unwrap();
        assert!(compiled.contains("0.25"));
        assert!(!compiled.contains("import "));
    }

    #[test]
    fn package_imports_load_registered_asset_sources_without_preloading() {
        let directory = Dir::default();
        directory.insert_asset(
            Path::new("root.wesl"),
            b"import library::shared::value; @fragment fn fragment() -> @location(0) vec4<f32> { return vec4(value()); }".as_slice(),
        );
        directory.insert_asset(
            Path::new("shared.wesl"),
            b"fn value() -> f32 { return f32(constants::LIB_VALUE); }".as_slice(),
        );
        let mut app = shader_app(directory.clone());
        let server = app.world().resource::<AssetServer>().clone();
        let handle = server.load::<Shader>("scoped://root.wesl");
        let decoy = server.load::<Shader>("scoped://shared.wesl");
        until(&mut app, |_| {
            server.is_loaded_with_dependencies(handle.id())
                && server.is_loaded_with_dependencies(decoy.id())
        });
        let shaders = app.world().resource::<Assets<Shader>>();
        let root = shaders.get(&handle).unwrap();
        assert_eq!(root.file_dependencies.len(), 1);
        let dependency = root.file_dependencies[0].clone();
        assert_eq!(
            shaders.get(&dependency).unwrap().path,
            "library://shared.wesl"
        );
        let mut cache = crate::ShaderCache::new((), |_, source, _| match source {
            crate::ShaderCacheSource::Wgsl(source) => Ok(source),
            _ => unreachable!(),
        });
        for (id, shader) in shaders.iter() {
            let mut shader = shader.clone();
            if id == dependency.id() {
                shader
                    .shader_defs
                    .push(ShaderDefVal::UInt("LIB_VALUE".into(), 3));
            } else if id == decoy.id() {
                shader
                    .shader_defs
                    .push(ShaderDefVal::UInt("LIB_VALUE".into(), 99));
            }
            cache.set_shader(id, shader);
        }
        let compiled = cache.get(0, handle.id(), &[]).unwrap();
        assert!(compiled.contains("3"));
        assert!(!compiled.contains("99"));
        directory.insert_asset(
            Path::new("shared.wesl"),
            b"fn value() -> f32 { return 0.5; }".as_slice(),
        );
        server.reload("library://shared.wesl");
        until(&mut app, |app| {
            app.world()
                .resource::<Assets<Shader>>()
                .get(&dependency)
                .is_some_and(|shader| shader.source.as_str().contains("0.5"))
        });
        let updated = app
            .world()
            .resource::<Assets<Shader>>()
            .get(&dependency)
            .unwrap()
            .clone();
        assert!(cache.set_shader(dependency.id(), updated).contains(&0));
        assert!(cache.get(0, handle.id(), &[]).unwrap().contains("0.5"));
    }
}

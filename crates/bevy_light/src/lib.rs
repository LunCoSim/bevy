//! Provides component types for lighting a bevy scene. This includes the usual
//! directional, point, and spot lights, as well as light probes, atmosphere,
//! other volumetrics, and shadow configuration.

extern crate alloc;

use bevy_app::{App, Plugin, PostUpdate, Update};
use bevy_asset::{AssetApp, AssetEventSystems};
use bevy_camera::{
    primitives::{Aabb, CascadesFrusta, CubemapFrusta, Frustum, Sphere},
    visibility::{
        CascadesVisibleEntities, CubemapVisibleEntities, InheritedVisibility, NoCpuCulling,
        NoFrustumCulling, RenderLayers, ViewVisibility, VisibilityRange, VisibilitySystems,
        VisibleEntities, VisibleEntityRanges, VisibleMeshEntities,
    },
    Camera, Camera3d, CameraUpdateSystems, Projection, RenderTarget, ShadowLodOrigin,
};
use bevy_ecs::{
    entity::{EntityHashMap, EntityHashSet},
    prelude::*,
    system::QueryLens,
};
#[cfg(feature = "bevy_gizmos")]
use bevy_gizmos::frustum::FrustumGizmoSystems;
use bevy_log::warn_once;
use bevy_math::{DMat4, DVec2, DVec4, UVec2, Vec3A};
use bevy_mesh::Mesh3d;
use bevy_reflect::prelude::*;
use bevy_transform::{components::GlobalTransform, TransformSystems};
use bevy_utils::Parallel;
use core::any::TypeId;
use smallvec::{smallvec, SmallVec};

pub mod cluster;
use cluster::assign::assign_objects_to_clusters;
pub use cluster::ClusteredDecal;
mod ambient_light;
use alloc::sync::Arc;
pub use ambient_light::{AmbientLight, GlobalAmbientLight};
use bevy_camera::visibility::{reconcile_sorted_visibility, SetViewVisibility};

mod probe;
pub use probe::{
    automatically_add_parallax_correction_components, AtmosphereEnvironmentMapLight,
    EnvironmentMapLight, GeneratedEnvironmentMapLight, IrradianceVolume, LightProbe,
    ParallaxCorrection, Skybox,
};
pub mod atmosphere;
pub use atmosphere::Atmosphere;
mod volumetric;
pub use volumetric::{FogVolume, VolumetricFog, VolumetricLight};
pub mod cascade;
use cascade::{build_directional_light_cascades, Cascade};
pub use cascade::{CascadeShadowConfig, CascadeShadowConfigBuilder, Cascades};
mod point_light;
pub use point_light::{
    update_point_light_frusta, PointLight, PointLightShadowMap, PointLightTexture,
};
mod spot_light;
pub use spot_light::{
    orthonormalize, spot_light_clip_from_view, spot_light_world_from_view,
    update_spot_light_frusta, SpotLight, SpotLightTexture,
};
mod directional_light;
pub use directional_light::{
    update_directional_light_frusta, DirectionalLight, DirectionalLightShadowMap,
    DirectionalLightTexture, SunDisk,
};
mod rect_light;
pub use rect_light::RectLight;
/// Provides gizmo drawing for visualizing light positions.
#[cfg(feature = "bevy_gizmos")]
pub mod gizmos;

/// The light prelude.
///
/// This includes the most common types in this crate, re-exported for your convenience.
pub mod prelude {
    #[doc(hidden)]
    pub use crate::{
        light_consts, AmbientLight, DirectionalLight, EnvironmentMapLight,
        GeneratedEnvironmentMapLight, GlobalAmbientLight, LightProbe, PointLight, RectLight,
        SpotLight,
    };

    #[doc(hidden)]
    #[cfg(feature = "bevy_gizmos")]
    pub use crate::gizmos::{LightGizmoColor, LightGizmoConfigGroup, ShowLightGizmo};
}

use crate::{
    atmosphere::{extract_chromatic_phase_textures, ScatteringMedium},
    cluster::{add_light_probe_and_decal_aabbs, ClusterVisibilityClass, Clusters},
    directional_light::validate_shadow_map_size,
    point_light::update_point_light_bounding_spheres,
    spot_light::update_spot_light_bounding_spheres,
};

/// Constants for operating with the light units: lumens, and lux.
pub mod light_consts {
    /// Approximations for converting the wattage of lamps to lumens.
    ///
    /// The **lumen** (symbol: **lm**) is the unit of [luminous flux], a measure
    /// of the total quantity of [visible light] emitted by a source per unit of
    /// time, in the [International System of Units] (SI).
    ///
    /// For more information, see [wikipedia](https://en.wikipedia.org/wiki/Lumen_(unit))
    ///
    /// [luminous flux]: https://en.wikipedia.org/wiki/Luminous_flux
    /// [visible light]: https://en.wikipedia.org/wiki/Visible_light
    /// [International System of Units]: https://en.wikipedia.org/wiki/International_System_of_Units
    pub mod lumens {
        /// The conversion factor used to determine how many lumens a typical LED light of a given wattage produces.
        pub const LUMENS_PER_LED_WATTS: f32 = 90.0;
        /// The conversion factor used to determine how many lumens a typical incandescent light of a given wattage produces.
        pub const LUMENS_PER_INCANDESCENT_WATTS: f32 = 13.8;
        /// The conversion factor used to determine how many lumens a typical halogen light of a given wattage produces.
        pub const LUMENS_PER_HALOGEN_WATTS: f32 = 19.8;
        /// 1,000,000 lumens is a very large "cinema light" capable of registering brightly at Bevy's
        /// default [`bevy_camera::Exposure::BLENDER`] exposure level. For "indoor lighting" with a lower exposure,
        /// this would be way too bright.
        pub const VERY_LARGE_CINEMA_LIGHT: f32 = 1_000_000.0;
    }

    /// Predefined for lux values in several locations.
    ///
    /// The **lux** (symbol: **lx**) is the unit of [illuminance], or [luminous flux] per unit area,
    /// in the [International System of Units] (SI). It is equal to one lumen per square meter.
    ///
    /// For more information, see [wikipedia](https://en.wikipedia.org/wiki/Lux)
    ///
    /// [illuminance]: https://en.wikipedia.org/wiki/Illuminance
    /// [luminous flux]: https://en.wikipedia.org/wiki/Luminous_flux
    /// [International System of Units]: https://en.wikipedia.org/wiki/International_System_of_Units
    pub mod lux {
        /// The amount of light (lux) in a moonless, overcast night sky. (starlight)
        pub const MOONLESS_NIGHT: f32 = 0.0001;
        /// The amount of light (lux) during a full moon on a clear night.
        pub const FULL_MOON_NIGHT: f32 = 0.05;
        /// The amount of light (lux) during the dark limit of civil twilight under a clear sky.
        pub const CIVIL_TWILIGHT: f32 = 3.4;
        /// The amount of light (lux) in family living room lights.
        pub const LIVING_ROOM: f32 = 50.;
        /// The amount of light (lux) in an office building's hallway/toilet lighting.
        pub const HALLWAY: f32 = 80.;
        /// The amount of light (lux) in very dark overcast day
        pub const DARK_OVERCAST_DAY: f32 = 100.;
        /// The amount of light (lux) in an office.
        pub const OFFICE: f32 = 320.;
        /// The amount of light (lux) during sunrise or sunset on a clear day.
        pub const CLEAR_SUNRISE: f32 = 400.;
        /// The amount of light (lux) on an overcast day; typical TV studio lighting
        pub const OVERCAST_DAY: f32 = 1000.;
        /// The amount of light (lux) from ambient daylight (not direct sunlight).
        /// This is the default for [`DirectionalLight`](crate::DirectionalLight)s in Bevy.
        pub const AMBIENT_DAYLIGHT: f32 = 10_000.;
        /// The amount of light (lux) in full daylight (not direct sun).
        pub const FULL_DAYLIGHT: f32 = 20_000.;
        /// The amount of light (lux) in direct sunlight.
        pub const DIRECT_SUNLIGHT: f32 = 100_000.;
        /// The amount of light (lux) of raw sunlight, not filtered by the atmosphere.
        pub const RAW_SUNLIGHT: f32 = 130_000.;
    }
}

/// Sets up all the light visibility and clustering infrastructure needed for rendering lights.
#[derive(Default)]
pub struct LightPlugin;

impl Plugin for LightPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GlobalAmbientLight>()
            .init_resource::<DirectionalLightShadowMap>()
            .init_resource::<PointLightShadowMap>()
            .init_asset::<ScatteringMedium>()
            .register_required_components::<Camera3d, Clusters>()
            .configure_sets(
                PostUpdate,
                SimulationLightSystems::CheckLightVisibility
                    .ambiguous_with(SimulationLightSystems::CheckLightVisibility),
            )
            .configure_sets(
                PostUpdate,
                SimulationLightSystems::AssignLightsToClusters
                    .before(bevy_app::TransformGizmoRenderStep),
            )
            .add_systems(Update, automatically_add_parallax_correction_components)
            .add_systems(
                PostUpdate,
                (
                    validate_shadow_map_size.before(build_directional_light_cascades),
                    assign_objects_to_clusters
                        .in_set(SimulationLightSystems::AssignLightsToClusters)
                        .after(TransformSystems::Propagate)
                        .after(VisibilitySystems::CheckVisibility)
                        .after(CameraUpdateSystems),
                    update_directional_light_frusta
                        .in_set(SimulationLightSystems::UpdateLightFrusta)
                        // This must run after CheckVisibility because it relies on `ViewVisibility`
                        .after(VisibilitySystems::CheckVisibility)
                        .after(TransformSystems::Propagate)
                        .after(SimulationLightSystems::UpdateDirectionalLightCascades)
                        // We assume that no entity will be both a directional light and a spot light,
                        // so these systems will run independently of one another.
                        // FIXME: Add an archetype invariant for this https://github.com/bevyengine/bevy/issues/1481.
                        .ambiguous_with(update_spot_light_frusta),
                    update_point_light_frusta
                        .in_set(SimulationLightSystems::UpdateLightFrusta)
                        .after(TransformSystems::Propagate)
                        .after(SimulationLightSystems::AssignLightsToClusters),
                    #[cfg(feature = "bevy_gizmos")]
                    update_spot_light_frusta
                        .in_set(SimulationLightSystems::UpdateLightFrusta)
                        .before(FrustumGizmoSystems)
                        .after(TransformSystems::Propagate)
                        .after(SimulationLightSystems::AssignLightsToClusters),
                    #[cfg(not(feature = "bevy_gizmos"))]
                    update_spot_light_frusta
                        .in_set(SimulationLightSystems::UpdateLightFrusta)
                        .after(TransformSystems::Propagate)
                        .after(SimulationLightSystems::AssignLightsToClusters),
                    (
                        check_dir_light_mesh_visibility,
                        check_point_light_mesh_visibility,
                    )
                        .in_set(SimulationLightSystems::CheckLightVisibility)
                        .after(VisibilitySystems::CalculateBounds)
                        .after(TransformSystems::Propagate)
                        .after(SimulationLightSystems::UpdateLightFrusta)
                        // Lights can "see" entities and mark them as visible. This is done to
                        // correctly render shadows for entities that are not in view of a camera,
                        // but must be renderable to cast shadows. Because of this, we need to check
                        // entity visibility and mark as visible before they can be hidden.
                        .after(VisibilitySystems::CheckVisibility)
                        .before(VisibilitySystems::MarkNewlyHiddenEntitiesInvisible),
                    (
                        update_point_light_bounding_spheres.after(TransformSystems::Propagate),
                        update_spot_light_bounding_spheres.after(TransformSystems::Propagate),
                        add_light_probe_and_decal_aabbs,
                    )
                        .in_set(SimulationLightSystems::UpdateBounds)
                        .before(VisibilitySystems::UpdateFrusta),
                    build_directional_light_cascades
                        .in_set(SimulationLightSystems::UpdateDirectionalLightCascades)
                        .after(TransformSystems::Propagate)
                        .before(bevy_app::TransformGizmoRenderStep)
                        .after(CameraUpdateSystems),
                    extract_chromatic_phase_textures.after(AssetEventSystems),
                ),
            );

        #[cfg(feature = "bevy_gizmos")]
        app.add_plugins(gizmos::LightGizmoPlugin);
    }
}

/// A convenient alias for `Or<(With<PointLight>, With<SpotLight>,
/// With<DirectionalLight>, With<RectLight>)>`, for use with [`bevy_camera::visibility::VisibleEntities`].
pub type WithLight = Or<(
    With<PointLight>,
    With<SpotLight>,
    With<DirectionalLight>,
    With<RectLight>,
)>;

/// Add this component to make a [`Mesh3d`] not cast shadows.
#[derive(Debug, Component, Reflect, Default, Clone, PartialEq)]
#[reflect(Component, Default, Debug, Clone, PartialEq)]
pub struct NotShadowCaster;
/// Add this component to make a [`Mesh3d`] not receive shadows.
///
/// **Note:** If you're using diffuse transmission, setting [`NotShadowReceiver`] will
/// cause both “regular” shadows as well as diffusely transmitted shadows to be disabled,
/// even when [`TransmittedShadowReceiver`] is being used.
#[derive(Debug, Component, Reflect, Default, Clone)]
#[reflect(Component, Default, Debug)]
pub struct NotShadowReceiver;
/// Add this component to make a [`Mesh3d`] using a PBR material with `StandardMaterial::diffuse_transmission > 0.0`
/// receive shadows on its diffuse transmission lobe. (i.e. its “backside”)
///
/// Not enabled by default, as it requires carefully setting up `StandardMaterial::thickness`
/// (and potentially even baking a thickness texture!) to match the geometry of the mesh, in order to avoid self-shadow artifacts.
///
/// **Note:** Using [`NotShadowReceiver`] overrides this component.
#[derive(Debug, Component, Reflect, Default, Clone)]
#[reflect(Component, Default, Debug)]
pub struct TransmittedShadowReceiver;

/// Add this component to a [`Camera3d`]
/// to control how to anti-alias shadow edges.
///
/// The different modes use different approaches to
/// [Percentage Closer Filtering](https://developer.nvidia.com/gpugems/gpugems/part-ii-lighting-and-shadows/chapter-11-shadow-map-antialiasing).
#[derive(Debug, Component, Reflect, Clone, Copy, PartialEq, Eq, Default)]
#[reflect(Component, Default, Debug, PartialEq, Clone)]
pub enum ShadowFilteringMethod {
    /// Hardware 2x2.
    ///
    /// Fast but poor quality.
    Hardware2x2,
    /// Phase-correct 3×3 tent on directional and spot maps. Cubemaps use hardware comparison.
    Tent3x3,
    /// Approximates a fixed Gaussian blur, good when TAA isn't in use.
    ///
    /// Good quality, good performance.
    ///
    /// For directional and spot lights, this uses a [method by Ignacio Castaño
    /// for *The Witness*] using 9 samples and smart filtering to achieve the same
    /// as a regular 5x5 filter kernel.
    ///
    /// [method by Ignacio Castaño for *The Witness*]: https://web.archive.org/web/20230210095515/http://the-witness.net/news/2013/09/shadow-mapping-summary-part-1/
    #[default]
    Gaussian,
    /// A randomized filter that varies over time, good when TAA is in use.
    ///
    /// Good quality when used with `TemporalAntiAliasing`
    /// and good performance.
    ///
    /// For directional and spot lights, this uses a [method by Jorge Jimenez for
    /// *Call of Duty: Advanced Warfare*] using 8 samples in spiral pattern,
    /// randomly-rotated by interleaved gradient noise with spatial variation.
    ///
    /// [method by Jorge Jimenez for *Call of Duty: Advanced Warfare*]: https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/
    Temporal,
}

/// Main-world identity of the directional light selected for a material's direct-light response.
#[derive(Resource, Clone, Copy, Default, PartialEq, Eq)]
pub struct MaterialResponseLight(pub Option<Entity>);

/// System sets used to run light-related systems.
#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub enum SimulationLightSystems {
    /// The set that adds AABBs and bounding spheres to clustered objects.
    UpdateBounds,
    /// After this set, all lights have been clustered.
    AssignLightsToClusters,
    /// After this set, all directional light cascades are up to date.
    UpdateDirectionalLightCascades,
    /// After this set, the frusta of shadow-casting point lights, spot lights, and directional lights are up to date.
    UpdateLightFrusta,
    /// System order ambiguities between systems in this set are ignored:
    /// the order of systems within this set is irrelevant, as the various visibility-checking systems
    /// assumes that their operations are irreversible during the frame.
    CheckLightVisibility,
}

/// Updates the visibility for [`DirectionalLight`]s so that shadow map
/// rendering can work.
///
/// This only processes entities without [`NoCpuCulling`]. Entities with
/// [`NoCpuCulling`] receive no view-specific processing in the main world.
pub fn check_dir_light_mesh_visibility(
    mut commands: Commands,
    mut directional_lights: Query<
        (
            Entity,
            &DirectionalLight,
            &CascadeShadowConfig,
            &Cascades,
            &CascadesFrusta,
            &mut CascadesVisibleEntities,
            Option<&RenderLayers>,
            &ViewVisibility,
        ),
        Without<SpotLight>,
    >,
    camera_views: Query<(
        &Camera,
        &Projection,
        &GlobalTransform,
        Option<&ShadowFilteringMethod>,
    )>,
    visible_entity_query: Query<
        (
            Entity,
            &InheritedVisibility,
            Option<&RenderLayers>,
            Option<&Aabb>,
            Option<&GlobalTransform>,
            Has<VisibilityRange>,
            Has<NoFrustumCulling>,
        ),
        (
            Without<NotShadowCaster>,
            Without<DirectionalLight>,
            Without<NoCpuCulling>,
            With<Mesh3d>,
        ),
    >,
    changed: Query<
        Entity,
        (
            With<Mesh3d>,
            With<InheritedVisibility>,
            Or<(
                Changed<Mesh3d>,
                Changed<InheritedVisibility>,
                Changed<RenderLayers>,
                Changed<Aabb>,
                Changed<GlobalTransform>,
                Added<NotShadowCaster>,
                Added<DirectionalLight>,
                Added<NoCpuCulling>,
                Added<NoFrustumCulling>,
                With<VisibilityRange>,
            )>,
        ),
    >,
    (
        mut removed_mesh,
        mut removed_inherited,
        mut removed_layers,
        mut removed_aabb,
        mut removed_transform,
        mut removed_not_caster,
        mut removed_light,
        mut removed_no_cpu,
        mut removed_no_frustum,
        mut removed_range,
    ): (
        RemovedComponents<Mesh3d>,
        RemovedComponents<InheritedVisibility>,
        RemovedComponents<RenderLayers>,
        RemovedComponents<Aabb>,
        RemovedComponents<GlobalTransform>,
        RemovedComponents<NotShadowCaster>,
        RemovedComponents<DirectionalLight>,
        RemovedComponents<NoCpuCulling>,
        RemovedComponents<NoFrustumCulling>,
        RemovedComponents<VisibilityRange>,
    ),
    visible_entity_ranges: Option<Res<VisibleEntityRanges>>,
    mut defer_visible_entities_queue: Local<Parallel<Vec<Entity>>>,
    mut view_visible_entities_queue: Local<Parallel<Vec<Vec<Entity>>>>,
    mut cache: Local<EntityHashMap<EntityHashMap<DirectionalVisibilityView>>>,
    mut dirty: Local<Vec<Entity>>,
    mut decisions: Local<Vec<(Entity, DirectionalVisibilityDecision)>>,
    mut scratch: Local<Vec<Entity>>,
) {
    let visible_entity_ranges = visible_entity_ranges.as_deref();
    dirty.clear();
    dirty.extend(
        removed_mesh
            .read()
            .chain(removed_inherited.read())
            .chain(removed_layers.read())
            .chain(removed_aabb.read())
            .chain(removed_transform.read())
            .chain(removed_not_caster.read())
            .chain(removed_light.read())
            .chain(removed_no_cpu.read())
            .chain(removed_no_frustum.read())
            .chain(removed_range.read()),
    );
    let mut collected = false;
    let mut publication = Vec::new();
    cache.retain(|light, _| directional_lights.get(*light).is_ok());

    for (
        light,
        directional_light,
        cascade_config,
        cascades,
        frusta,
        mut visible_entities,
        maybe_view_mask,
        light_view_visibility,
    ) in &mut directional_lights
    {
        if !directional_light.shadow_maps_enabled || !light_view_visibility.get() {
            visible_entities.entities.clear();
            cache.remove(&light);
            continue;
        }
        let view_mask = maybe_view_mask.unwrap_or_default();
        let cached_views = cache.entry(light).or_default();
        if visible_entities.is_changed() {
            cached_views.clear();
        }
        cached_views.retain(|view, _| frusta.frusta.contains_key(view));
        if visible_entities
            .entities
            .keys()
            .any(|view| !frusta.frusta.contains_key(view))
        {
            visible_entities
                .entities
                .retain(|view, _| frusta.frusta.contains_key(view));
        }
        for (view, view_frusta) in &frusta.frusta {
            let receiver_bounds = camera_views
                .get(*view)
                .ok()
                .and_then(|(camera, projection, transform, filtering)| {
                    if !directional_shadow_receiver_culling_supported(directional_light) {
                        return None;
                    }
                    let cascade_views = cascades.cascades.get(view)?;
                    if !camera.is_active
                        || camera.sub_camera_view.is_some()
                        || cascade_views.len() != view_frusta.len()
                        || cascade_config.bounds.len() != view_frusta.len()
                    {
                        return None;
                    }

                    let viewport_size = camera.physical_viewport_size()?;
                    if viewport_size.x == 0 || viewport_size.y == 0 {
                        return None;
                    }

                    let filter_radius =
                        directional_shadow_filter_radius(filtering.copied().unwrap_or_default())?;
                    let overlap_factor = 1.0 - cascade_config.overlap_proportion;
                    let near_bounds = [cascade_config.minimum_distance]
                        .into_iter()
                        .chain(cascade_config.bounds.iter().map(|far| overlap_factor * far));
                    Some(
                        near_bounds
                            .zip(&cascade_config.bounds)
                            .zip(cascade_views)
                            .map(|((near, far), cascade)| {
                                directional_shadow_receiver_bounds(
                                    projection,
                                    transform,
                                    cascade,
                                    near,
                                    *far,
                                    viewport_size,
                                    filter_radius,
                                    directional_light.receiver_normal_bias(),
                                )
                            })
                            .collect::<SmallVec<[Option<DirectionalShadowReceiverBounds>; 4]>>(),
                    )
                })
                .unwrap_or_else(|| {
                    let mut bounds =
                        SmallVec::<[Option<DirectionalShadowReceiverBounds>; 4]>::new();
                    bounds.resize(view_frusta.len(), None);
                    bounds
                });

            let stable = cached_views.get(view).is_some_and(|old| {
                old.layers == *view_mask
                    && old.frusta.len() == view_frusta.len()
                    && old.frusta.iter().zip(view_frusta).all(|(a, b)| a.0 == b.0)
                    && old.receiver_bounds == receiver_bounds
                    && visible_entities
                        .entities
                        .get(view)
                        .is_some_and(|lists| lists.len() == view_frusta.len())
            });
            if stable && !collected {
                dirty.extend(changed.iter());
                dirty.sort_unstable();
                dirty.dedup();
                collected = true;
            }
            // For widespread changes the native parallel archetype traversal
            // costs less than sparse random lookups and merging retained lists.
            if stable && dirty.len() <= visible_entity_query.iter().len() / 4 {
                let old = cached_views.get_mut(view).unwrap();
                if !dirty.is_empty() {
                    decisions.clear();
                    for &entity in dirty.iter() {
                        let mut decision = DirectionalVisibilityDecision::default();
                        if let Ok((_, inherited, layers, aabb, transform, range, no_frustum)) =
                            visible_entity_query.get(entity)
                        {
                            decision.visible = visit_directional_caster(
                                entity,
                                inherited,
                                layers,
                                aabb,
                                transform,
                                range,
                                no_frustum,
                                *view,
                                view_mask,
                                view_frusta,
                                &receiver_bounds,
                                visible_entity_ranges,
                                |index| decision.cascades.push(index),
                            );
                        }
                        decisions.push((entity, decision));
                    }
                    reconcile_sorted_visibility(
                        Arc::make_mut(&mut old.union),
                        &decisions,
                        |decision| usize::from(decision.visible),
                        &mut scratch,
                    );
                    let lists = visible_entities.entities.get_mut(view).unwrap();
                    for (index, list) in lists.iter_mut().enumerate() {
                        reconcile_sorted_visibility(
                            &mut list.entities,
                            &decisions,
                            |decision| usize::from(decision.cascades.contains(&index)),
                            &mut scratch,
                        );
                    }
                }
                publication.push(Arc::clone(&old.union));
                continue;
            }

            // Native full culling path for changed views and broadly dirty inventories.
            for thread_queue in view_visible_entities_queue.iter_mut() {
                thread_queue.resize(view_frusta.len(), Vec::default());
            }
            visible_entity_query.par_iter().for_each_init(
                || {
                    let mut entities = view_visible_entities_queue.borrow_local_mut();
                    entities.resize(view_frusta.len(), Vec::default());
                    (defer_visible_entities_queue.borrow_local_mut(), entities)
                },
                |(union, lists),
                 (entity, inherited, layers, aabb, transform, range, no_frustum)| {
                    if visit_directional_caster(
                        entity,
                        inherited,
                        layers,
                        aabb,
                        transform,
                        range,
                        no_frustum,
                        *view,
                        view_mask,
                        view_frusta,
                        &receiver_bounds,
                        visible_entity_ranges,
                        |index| lists[index].push(entity),
                    ) {
                        union.push(entity);
                    }
                },
            );
            let old = cached_views.entry(*view).or_default();
            old.layers.clone_from(view_mask);
            old.frusta.clear();
            old.frusta.extend_from_slice(view_frusta);
            old.receiver_bounds = receiver_bounds;
            let union = Arc::make_mut(&mut old.union);
            union.clear();
            for thread_union in defer_visible_entities_queue.iter_mut() {
                union.append(thread_union);
            }
            union.sort_unstable();
            let lists = visible_entities.entities.entry(*view).or_default();
            lists.resize(view_frusta.len(), Default::default());
            for (index, list) in lists.iter_mut().enumerate() {
                list.entities.clear();
                for thread_lists in view_visible_entities_queue.iter_mut() {
                    list.entities.append(&mut thread_lists[index]);
                }
                list.shrink();
                list.entities.sort_unstable();
            }
            publication.push(Arc::clone(&old.union));
        }
    }

    // ViewVisibility is a per-frame union. Borrow retained immutable membership
    // through a shared lease instead of rebuilding a per-caster deferred queue.
    if !publication.is_empty() {
        commands.queue(move |world: &mut World| {
            let mut query = world.query::<&mut ViewVisibility>();
            for entities in publication {
                let mut iter = query.iter_many_mut(world, entities.iter()).matched();
                while let Some(mut visibility) = iter.fetch_next() {
                    visibility.set_visible();
                }
            }
        });
    }
}

#[doc(hidden)]
#[derive(Default)]
pub struct DirectionalVisibilityView {
    layers: RenderLayers,
    frusta: SmallVec<[Frustum; 4]>,
    receiver_bounds: SmallVec<[Option<DirectionalShadowReceiverBounds>; 4]>,
    union: Arc<Vec<Entity>>,
}

#[doc(hidden)]
#[derive(Default)]
pub struct DirectionalVisibilityDecision {
    visible: bool,
    cascades: SmallVec<[usize; 4]>,
}

#[inline]
fn visit_directional_caster(
    entity: Entity,
    inherited: &InheritedVisibility,
    layers: Option<&RenderLayers>,
    aabb: Option<&Aabb>,
    transform: Option<&GlobalTransform>,
    range: bool,
    no_frustum: bool,
    view: Entity,
    view_mask: &RenderLayers,
    frusta: &[Frustum],
    bounds: &[Option<DirectionalShadowReceiverBounds>],
    ranges: Option<&VisibleEntityRanges>,
    mut selected: impl FnMut(usize),
) -> bool {
    if !inherited.get() || !view_mask.intersects(layers.unwrap_or_default()) {
        return false;
    }
    if range && ranges.is_some_and(|ranges| !ranges.entity_is_in_range_of_view(entity, view)) {
        return false;
    }
    if let (Some(aabb), Some(transform)) = (aabb, transform) {
        let mut visible = false;
        let affine = transform.affine();
        for (index, frustum) in frusta.iter().enumerate() {
            // Directional casters can be upstream of the receiver's near plane.
            if !no_frustum
                && (!frustum.intersects_obb(aabb, &affine, false, true)
                    || bounds[index].is_some_and(|bounds| {
                        aabb_is_outside_directional_shadow_receiver_bounds(aabb, transform, &bounds)
                    }))
            {
                continue;
            }
            visible = true;
            selected(index);
        }
        visible
    } else {
        for index in 0..frusta.len() {
            selected(index);
        }
        true
    }
}

/// Conservative screen-receiver bounds in directional-light clip space.
///
/// Directional casters may be arbitrarily far upstream of a receiver, so the
/// bounds constrain only light-space X/Y. Z is deliberately unbounded here.
#[derive(Clone, Copy, PartialEq)]
struct DirectionalShadowReceiverBounds {
    clip_from_world: DMat4,
    axes: [DirectionalShadowReceiverAxis; 8],
    vertex_count: u8,
    min: DVec2,
    max: DVec2,
    padding: DVec2,
}

#[derive(Clone, Copy, Default, PartialEq)]
struct DirectionalShadowReceiverAxis {
    direction: DVec2,
    min: f64,
    max: f64,
}

/// Conservative support in shadow texels for a fixed directional filter.
pub fn directional_shadow_filter_radius(method: ShadowFilteringMethod) -> Option<f64> {
    match method {
        ShadowFilteringMethod::Hardware2x2 => Some(1.0),
        ShadowFilteringMethod::Gaussian => Some(3.0),
        ShadowFilteringMethod::Tent3x3 => Some(2.0),
        ShadowFilteringMethod::Temporal => None,
    }
}

fn directional_shadow_receiver_culling_supported(_light: &DirectionalLight) -> bool {
    #[cfg(feature = "experimental_pbr_pcss")]
    if _light.soft_shadow_size.is_some_and(|size| size > 0.0) {
        return false;
    }

    true
}

/// Projects the full camera frustum slice into one cascade's light-space plane.
/// Filter and normal-bias footprints are included so this test cannot remove a
/// caster whose shadow sample can reach a visible receiver.
fn directional_shadow_receiver_bounds(
    projection: &Projection,
    world_from_view: &GlobalTransform,
    cascade: &Cascade,
    near: f32,
    far: f32,
    viewport_size: UVec2,
    filter_radius: f64,
    shadow_normal_bias: f32,
) -> Option<DirectionalShadowReceiverBounds> {
    let clip_from_world = cascade.clip_from_world.as_dmat4();
    let world_from_view = world_from_view.to_matrix().as_dmat4();
    if !near.is_finite()
        || !far.is_finite()
        || near < 0.0
        || near >= far
        || !filter_radius.is_finite()
        || filter_radius < 0.0
        || !shadow_normal_bias.is_finite()
        || !cascade.texel_size.is_finite()
        || cascade.texel_size <= 0.0
        || !world_from_view.is_finite()
        || !clip_from_world.is_finite()
        || clip_from_world.row(3) != DVec4::W
    {
        return None;
    }

    let mut min = DVec2::splat(f64::INFINITY);
    let mut max = DVec2::splat(f64::NEG_INFINITY);
    let mut numerical_padding = DVec2::ZERO;
    let mut points = SmallVec::<[DVec2; 8]>::new();
    let corners =
        frustum_corners_with_temporal_jitter_margin(projection, near, far, viewport_size)?;
    for corner in corners {
        let point = corner.as_dvec3().extend(1.0);
        let world = world_from_view * point;
        let clip = clip_from_world * world;
        let world_magnitude = world_from_view.abs() * point.abs();
        let clip_magnitude = clip_from_world.abs() * world_magnitude;
        let error = clip_magnitude.truncate().truncate() * (64.0 * f32::EPSILON as f64);
        let xy = clip.truncate().truncate();
        if !xy.is_finite() || !error.is_finite() {
            return None;
        }
        min = min.min(xy);
        max = max.max(xy);
        numerical_padding = numerical_padding.max(error);
        points.push(xy);
    }

    let (vertices, vertex_count) = directional_shadow_receiver_hull(points)?;
    let axes = directional_shadow_receiver_axes(&vertices[..usize::from(vertex_count)])?;

    let scale = DVec2::new(
        clip_from_world.row(0).truncate().length(),
        clip_from_world.row(1).truncate().length(),
    );
    let padding = scale
        * (f64::from(cascade.texel_size) * (filter_radius + f64::from(shadow_normal_bias.abs())));
    let padding = padding + numerical_padding;
    if !min.is_finite()
        || !max.is_finite()
        || min.x > max.x
        || min.y > max.y
        || !padding.is_finite()
        || padding.min_element() < 0.0
    {
        return None;
    }

    Some(DirectionalShadowReceiverBounds {
        clip_from_world,
        axes,
        vertex_count,
        min: min - padding,
        max: max + padding,
        padding,
    })
}

/// Returns the convex light-space silhouette of the projected receiver slice.
/// The projection of a convex frustum slice is the convex hull of its projected
/// corners; casters outside that hull cannot contribute to any receiver sample.
fn directional_shadow_receiver_hull(mut points: SmallVec<[DVec2; 8]>) -> Option<([DVec2; 8], u8)> {
    points.sort_unstable_by(|left, right| {
        left.x
            .total_cmp(&right.x)
            .then_with(|| left.y.total_cmp(&right.y))
    });
    points.dedup();
    if points.len() < 3 || points.iter().any(|point| !point.is_finite()) {
        return None;
    }

    let mut lower = SmallVec::<[DVec2; 8]>::new();
    for point in points.iter().copied() {
        while lower.len() >= 2
            && directional_shadow_hull_cross(lower[lower.len() - 2], lower[lower.len() - 1], point)
                <= 0.0
        {
            lower.pop();
        }
        lower.push(point);
    }

    let mut upper = SmallVec::<[DVec2; 8]>::new();
    for point in points.iter().rev().copied() {
        while upper.len() >= 2
            && directional_shadow_hull_cross(upper[upper.len() - 2], upper[upper.len() - 1], point)
                <= 0.0
        {
            upper.pop();
        }
        upper.push(point);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    if lower.len() < 3 || lower.len() > 8 {
        return None;
    }

    let mut vertices = [DVec2::ZERO; 8];
    vertices[..lower.len()].copy_from_slice(&lower);
    Some((vertices, lower.len() as u8))
}

fn directional_shadow_hull_cross(origin: DVec2, a: DVec2, b: DVec2) -> f64 {
    (a.x - origin.x) * (b.y - origin.y) - (a.y - origin.y) * (b.x - origin.x)
}

fn directional_shadow_receiver_axes(
    vertices: &[DVec2],
) -> Option<[DirectionalShadowReceiverAxis; 8]> {
    if !(3..=8).contains(&vertices.len()) || vertices.iter().any(|vertex| !vertex.is_finite()) {
        return None;
    }

    let mut axes = [DirectionalShadowReceiverAxis::default(); 8];
    for index in 0..vertices.len() {
        let edge = vertices[(index + 1) % vertices.len()] - vertices[index];
        let direction = DVec2::new(-edge.y, edge.x);
        if !direction.is_finite() || direction == DVec2::ZERO {
            return None;
        }

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for vertex in vertices {
            let projection = vertex.dot(direction);
            if !projection.is_finite() {
                return None;
            }
            min = min.min(projection);
            max = max.max(projection);
        }
        axes[index] = DirectionalShadowReceiverAxis {
            direction,
            min,
            max,
        };
    }

    Some(axes)
}

/// Expand each built-in frustum slice by half a physical pixel on every side.
/// Bevy applies `TemporalJitter` to the extracted projection, after main-world
/// visibility has run. This envelope includes every documented jitter offset
/// without coupling `bevy_light` to `bevy_render`. Custom projections and
/// sub-camera views keep native culling because their jitter envelope is not
/// defined by this projection contract.
fn frustum_corners_with_temporal_jitter_margin(
    projection: &Projection,
    near: f32,
    far: f32,
    viewport_size: UVec2,
) -> Option<[Vec3A; 8]> {
    if viewport_size.x == 0 || viewport_size.y == 0 {
        return None;
    }
    if !matches!(
        projection,
        Projection::Perspective(_) | Projection::Orthographic(_)
    ) {
        return None;
    }

    let mut corners = projection.get_frustum_corners(-near, -far);
    for plane_start in [0, 4] {
        let plane = &corners[plane_start..plane_start + 4];
        let mut min = bevy_math::Vec2::splat(f32::INFINITY);
        let mut max = bevy_math::Vec2::splat(f32::NEG_INFINITY);
        for corner in plane {
            min = min.min(corner.truncate());
            max = max.max(corner.truncate());
        }
        let span = max - min;
        if !min.is_finite() || !max.is_finite() || span.min_element() <= 0.0 {
            return None;
        }

        let margin = span / viewport_size.as_vec2() * 0.5;
        let center = (min + max) * 0.5;
        for corner in &mut corners[plane_start..plane_start + 4] {
            corner.x += if corner.x <= center.x {
                -margin.x
            } else {
                margin.x
            };
            corner.y += if corner.y <= center.y {
                -margin.y
            } else {
                margin.y
            };
        }
    }

    Some(corners)
}

/// Returns true only when the complete projected mesh AABB misses the receiver
/// hull. Invalid or non-affine bounds stay visible to preserve correctness.
fn aabb_is_outside_directional_shadow_receiver_bounds(
    aabb: &Aabb,
    model_to_world: &GlobalTransform,
    receiver: &DirectionalShadowReceiverBounds,
) -> bool {
    let center = aabb.center.as_dvec3();
    let half_extents = aabb.half_extents.as_dvec3();
    let model_to_world = model_to_world.to_matrix().as_dmat4();
    if !center.is_finite()
        || !half_extents.is_finite()
        || half_extents.x < 0.0
        || half_extents.y < 0.0
        || half_extents.z < 0.0
        || !model_to_world.is_finite()
    {
        return false;
    }

    let local_center = center.extend(1.0);
    let local_half_extents = half_extents.extend(0.0);
    let world_center = model_to_world * local_center;
    let world_center_magnitude = model_to_world.abs() * local_center.abs();
    let world_half_extents = model_to_world.abs() * local_half_extents;
    let clip_center = receiver.clip_from_world * world_center;
    let clip_center_magnitude = receiver.clip_from_world.abs() * world_center_magnitude;
    let clip_half_extents = receiver.clip_from_world.abs() * world_half_extents;
    let error = (clip_center_magnitude.truncate().truncate()
        + clip_half_extents.truncate().truncate())
        * (64.0 * f32::EPSILON as f64);
    let center = clip_center.truncate().truncate();
    let half_extents = clip_half_extents.truncate().truncate();
    if !center.is_finite() || !half_extents.is_finite() || !error.is_finite() {
        return false;
    }

    let min = center - half_extents - error;
    let max = center + half_extents + error;
    if max.x < receiver.min.x
        || min.x > receiver.max.x
        || max.y < receiver.min.y
        || min.y > receiver.max.y
    {
        return true;
    }

    for axis in &receiver.axes[..usize::from(receiver.vertex_count)] {
        let axis_abs = axis.direction.abs();
        let receiver_padding = receiver.padding.dot(axis_abs);
        let aabb_center = center.dot(axis.direction);
        let aabb_radius = half_extents.dot(axis_abs) + error.dot(axis_abs);
        if !receiver_padding.is_finite() || !aabb_center.is_finite() || !aabb_radius.is_finite() {
            return false;
        }

        let receiver_min = axis.min - receiver_padding;
        let receiver_max = axis.max + receiver_padding;
        let aabb_min = aabb_center - aabb_radius;
        let aabb_max = aabb_center + aabb_radius;
        if aabb_max < receiver_min || aabb_min > receiver_max {
            return true;
        }
    }

    false
}

type LocalCasterQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static InheritedVisibility,
        Option<&'static RenderLayers>,
        Option<&'static Aabb>,
        Option<&'static GlobalTransform>,
        Has<VisibilityRange>,
        Has<NoFrustumCulling>,
    ),
    (
        Without<NotShadowCaster>,
        Without<DirectionalLight>,
        Without<NoCpuCulling>,
        With<Mesh3d>,
        With<ViewVisibility>,
    ),
>;

#[derive(Default)]
struct LocalShadowView {
    center: Vec3A,
    range: f32,
    layers: RenderLayers,
    frusta: SmallVec<[Frustum; 6]>,
    origin: Option<Entity>,
    union: Arc<Vec<Entity>>,
}

impl LocalShadowView {
    fn matches(
        &self,
        sphere: &Sphere,
        layers: &RenderLayers,
        frusta: &[Frustum],
        origin: Option<Entity>,
    ) -> bool {
        self.center == sphere.center
            && self.range == sphere.radius
            && self.layers == *layers
            && self.origin == origin
            && self.frusta.len() == frusta.len()
            && self.frusta.iter().zip(frusta).all(|(a, b)| a.0 == b.0)
    }
}

struct LocalShadowJob {
    entity: Entity,
    point: bool,
    sphere: Sphere,
    layers: RenderLayers,
    frusta: SmallVec<[Frustum; 6]>,
}

#[derive(Default)]
struct LocalShadowScratch {
    queues: Parallel<Vec<Vec<(Entity, u8)>>>,
    decisions: Vec<(Entity, u8)>,
    merge: Vec<Entity>,
}

#[doc(hidden)]
#[derive(Default)]
pub struct LocalShadowVisibilityCache {
    points: EntityHashMap<LocalShadowView>,
    spots: EntityHashMap<LocalShadowView>,
    dirty: Vec<Entity>,
    jobs: Vec<LocalShadowJob>,
    scratch: LocalShadowScratch,
}

// A bit per shadow face. This uses the same layer, range, sphere and frustum
// tests for both sparse reconciliation and a full native parallel traversal.
fn local_caster_faces(
    entity: Entity,
    inherited: &InheritedVisibility,
    layers: Option<&RenderLayers>,
    aabb: Option<&Aabb>,
    transform: Option<&GlobalTransform>,
    has_range: bool,
    no_frustum: bool,
    view_mask: &RenderLayers,
    sphere: &Sphere,
    frusta: &[Frustum],
    origin: Option<Entity>,
    ranges: Option<&VisibleEntityRanges>,
) -> u8 {
    if !inherited.get() || !view_mask.intersects(layers.unwrap_or_default()) {
        return 0;
    }
    if has_range
        && ranges.is_some_and(|ranges| {
            origin.is_none_or(|origin| !ranges.entity_is_in_range_of_view(entity, origin))
        })
    {
        return 0;
    }
    if no_frustum || aabb.is_none() || transform.is_none() {
        return (1_u8 << frusta.len()) - 1;
    }
    let model_to_world = transform.unwrap().affine();
    let aabb = aabb.unwrap();
    if !sphere.intersects_obb(aabb, &model_to_world) {
        return 0;
    }
    frusta.iter().enumerate().fold(0, |mask, (index, frustum)| {
        if frustum.intersects_obb(aabb, &model_to_world, true, true) {
            mask | (1 << index)
        } else {
            mask
        }
    })
}

fn reconcile_local_shadow_view<'a>(
    view: &mut LocalShadowView,
    dirty: &[Entity],
    scratch: &mut LocalShadowScratch,
    sphere: &Sphere,
    layers: &RenderLayers,
    frusta: &[Frustum],
    origin: Option<Entity>,
    ranges: Option<&VisibleEntityRanges>,
    casters: &LocalCasterQuery,
    lists: impl Iterator<Item = &'a mut VisibleMeshEntities>,
) -> bool {
    let test = |(entity, inherited, mask, aabb, transform, range, no_frustum)| {
        local_caster_faces(
            entity, inherited, mask, aabb, transform, range, no_frustum, layers, sphere, frusta,
            origin, ranges,
        )
    };
    scratch.decisions.clear();
    scratch.decisions.extend(
        dirty
            .iter()
            .map(|&entity| (entity, casters.get(entity).map_or(0, test))),
    );
    let mut membership_changed = false;
    let union = Arc::make_mut(&mut view.union);
    if scratch
        .decisions
        .iter()
        .any(|(entity, mask)| union.binary_search(entity).is_ok() != (*mask != 0))
    {
        reconcile_sorted_visibility(
            union,
            &scratch.decisions,
            |mask| usize::from(*mask != 0),
            &mut scratch.merge,
        );
    }
    for (face, list) in lists.enumerate() {
        let bit = 1 << face;
        if scratch
            .decisions
            .iter()
            .any(|(entity, mask)| list.entities.binary_search(entity).is_ok() != (mask & bit != 0))
        {
            reconcile_sorted_visibility(
                &mut list.entities,
                &scratch.decisions,
                |mask| usize::from(mask & bit != 0),
                &mut scratch.merge,
            );
            membership_changed = true;
        }
    }
    membership_changed
}

fn publish_full_local_shadow_view<'a>(
    view: &mut LocalShadowView,
    job: &LocalShadowJob,
    origin: Option<Entity>,
    decisions: &[(Entity, u8)],
    lists: impl Iterator<Item = &'a mut VisibleMeshEntities>,
) -> bool {
    let mut membership_changed = false;
    let union = Arc::make_mut(&mut view.union);
    union.clear();
    union.extend(decisions.iter().map(|(entity, _)| *entity));
    for (face, list) in lists.enumerate() {
        let selected = || {
            decisions
                .iter()
                .filter_map(|(entity, mask)| (mask & (1 << face) != 0).then_some(*entity))
        };
        if !list.entities.iter().copied().eq(selected()) {
            list.entities.clear();
            list.entities.extend(selected());
            membership_changed = true;
        }
    }
    view.center = job.sphere.center;
    view.range = job.sphere.radius;
    view.layers.clone_from(&job.layers);
    view.frusta.clone_from(&job.frusta);
    view.origin = origin;
    membership_changed
}

/// Updates the visibility for [`PointLight`]s and [`SpotLight`]s so that shadow
/// map rendering can work. Retains exact memberships for unchanged light views
/// and reconciles changed casters, while republishing their per-frame visibility union.
/// Entities with [`NoCpuCulling`] receive no view-specific processing in the main world.
pub fn check_point_light_mesh_visibility(
    mut commands: Commands,
    visible_point_lights: Query<&VisibleEntities>,
    mut point_lights: Query<(
        Ref<PointLight>,
        &GlobalTransform,
        &CubemapFrusta,
        &mut CubemapVisibleEntities,
        Option<&RenderLayers>,
    )>,
    mut spot_lights: Query<(
        Ref<SpotLight>,
        &GlobalTransform,
        &Frustum,
        &mut VisibleMeshEntities,
        Option<&RenderLayers>,
    )>,
    casters: LocalCasterQuery,
    changed: Query<
        Entity,
        (
            With<Mesh3d>,
            With<InheritedVisibility>,
            Or<(
                Changed<Mesh3d>,
                Changed<InheritedVisibility>,
                Changed<RenderLayers>,
                Added<ViewVisibility>,
                Changed<Aabb>,
                Changed<GlobalTransform>,
                Added<NotShadowCaster>,
                Added<DirectionalLight>,
                Added<NoCpuCulling>,
                Added<NoFrustumCulling>,
                With<VisibilityRange>,
            )>,
        ),
    >,
    disabled: Query<
        Entity,
        (
            With<Mesh3d>,
            Added<bevy_ecs::entity_disabling::Disabled>,
            Allow<bevy_ecs::entity_disabling::Disabled>,
        ),
    >,
    (
        mut removed_mesh,
        mut removed_inherited,
        mut removed_layers,
        mut removed_aabb,
        mut removed_transform,
        mut removed_not_caster,
        mut removed_light,
        mut removed_no_cpu,
        mut removed_no_frustum,
        mut removed_range,
        mut removed_visibility,
        mut enabled,
    ): (
        RemovedComponents<Mesh3d>,
        RemovedComponents<InheritedVisibility>,
        RemovedComponents<RenderLayers>,
        RemovedComponents<Aabb>,
        RemovedComponents<GlobalTransform>,
        RemovedComponents<NotShadowCaster>,
        RemovedComponents<DirectionalLight>,
        RemovedComponents<NoCpuCulling>,
        RemovedComponents<NoFrustumCulling>,
        RemovedComponents<VisibilityRange>,
        RemovedComponents<ViewVisibility>,
        RemovedComponents<bevy_ecs::entity_disabling::Disabled>,
    ),
    mut camera_query: Query<(Entity, &RenderTarget), With<Camera>>,
    mut shadow_lod_origin_query: Query<Entity, With<ShadowLodOrigin>>,
    mut point_and_spot_light_query: Query<Entity, Or<(With<PointLight>, With<SpotLight>)>>,
    visible_entity_ranges: Option<Res<VisibleEntityRanges>>,
    mut cache: Local<LocalShadowVisibilityCache>,
    mut checked_lights: Local<EntityHashSet>,
) {
    let cache = &mut *cache;
    checked_lights.clear();
    cache.dirty.clear();
    cache.dirty.extend(
        removed_mesh
            .read()
            .chain(removed_inherited.read())
            .chain(removed_layers.read())
            .chain(removed_aabb.read())
            .chain(removed_transform.read())
            .chain(removed_not_caster.read())
            .chain(removed_light.read())
            .chain(removed_no_cpu.read())
            .chain(removed_no_frustum.read())
            .chain(removed_range.read())
            .chain(removed_visibility.read())
            .chain(enabled.read()),
    );
    let origin = get_shadow_lod_origin(
        camera_query.transmute_lens_filtered(),
        shadow_lod_origin_query.transmute_lens_filtered(),
        point_and_spot_light_query.transmute_lens_filtered(),
    );
    let ranges = visible_entity_ranges.as_deref();
    let mut collected = false;
    let sparse_limit = casters.iter().len() / 4;
    let mut publication = Vec::new();
    cache.jobs.clear();

    for visible_lights in &visible_point_lights {
        for &light_entity in visible_lights.get(TypeId::of::<ClusterVisibilityClass>()) {
            if !checked_lights.insert(light_entity) {
                continue;
            }
            if let Ok((light, transform, frusta, mut lists, layers)) =
                point_lights.get_mut(light_entity)
            {
                if light.shadow_maps_enabled {
                    let sphere = Sphere {
                        center: Vec3A::from(transform.translation()),
                        radius: light.range,
                    };
                    let layers = layers.unwrap_or_default();
                    let frusta = &frusta.frusta;
                    let stable = !light.is_added()
                        && !lists.is_changed()
                        && cache
                            .points
                            .get(&light_entity)
                            .is_some_and(|view| view.matches(&sphere, layers, frusta, origin));
                    if stable && !collected {
                        cache.dirty.extend(changed.iter().chain(disabled.iter()));
                        cache.dirty.sort_unstable();
                        cache.dirty.dedup();
                        collected = true;
                    }
                    if !stable || cache.dirty.len() > sparse_limit {
                        cache.jobs.push(LocalShadowJob {
                            entity: light_entity,
                            point: true,
                            sphere,
                            layers: layers.clone(),
                            frusta: SmallVec::from_slice(frusta),
                        });
                    } else {
                        let view = cache.points.entry(light_entity).or_default();
                        if !cache.dirty.is_empty()
                            && reconcile_local_shadow_view(
                                view,
                                &cache.dirty,
                                &mut cache.scratch,
                                &sphere,
                                layers,
                                frusta,
                                origin,
                                ranges,
                                &casters,
                                lists.bypass_change_detection().iter_mut(),
                            )
                        {
                            lists.set_changed();
                        }
                        publication.push(Arc::clone(&view.union));
                    }
                } else {
                    cache.points.remove(&light_entity);
                    cache.spots.remove(&light_entity);
                    continue;
                }
            }
            if let Ok((light, transform, frustum, mut list, layers)) =
                spot_lights.get_mut(light_entity)
            {
                if light.shadow_maps_enabled {
                    let sphere = Sphere {
                        center: Vec3A::from(transform.translation()),
                        radius: light.range,
                    };
                    let layers = layers.unwrap_or_default();
                    let frusta = core::slice::from_ref(frustum);
                    let stable = !light.is_added()
                        && !list.is_changed()
                        && cache
                            .spots
                            .get(&light_entity)
                            .is_some_and(|view| view.matches(&sphere, layers, frusta, origin));
                    if stable && !collected {
                        cache.dirty.extend(changed.iter().chain(disabled.iter()));
                        cache.dirty.sort_unstable();
                        cache.dirty.dedup();
                        collected = true;
                    }
                    if !stable || cache.dirty.len() > sparse_limit {
                        cache.jobs.push(LocalShadowJob {
                            entity: light_entity,
                            point: false,
                            sphere,
                            layers: layers.clone(),
                            frusta: SmallVec::from_slice(frusta),
                        });
                    } else {
                        let view = cache.spots.entry(light_entity).or_default();
                        if !cache.dirty.is_empty()
                            && reconcile_local_shadow_view(
                                view,
                                &cache.dirty,
                                &mut cache.scratch,
                                &sphere,
                                layers,
                                frusta,
                                origin,
                                ranges,
                                &casters,
                                core::iter::once(list.bypass_change_detection()),
                            )
                        {
                            list.set_changed();
                        }
                        publication.push(Arc::clone(&view.union));
                    }
                } else {
                    cache.spots.remove(&light_entity);
                }
            }
        }
    }
    // Moving lights share one parallel archetype traversal. Every caster is
    // borrowed once, and each worker retains distinct per-light output queues.
    // The original sphere/OBB/frustum predicates remain the final authority.
    if !cache.jobs.is_empty() {
        let jobs = &cache.jobs;
        let queues = &mut cache.scratch.queues;
        for thread in queues.iter_mut() {
            thread.resize_with(jobs.len(), Vec::new);
        }
        casters.par_iter().for_each_init(
            || {
                let mut thread = queues.borrow_local_mut();
                thread.resize_with(jobs.len(), Vec::new);
                thread
            },
            |thread, (entity, inherited, mask, aabb, transform, range, no_frustum)| {
                if !inherited.get()
                    || (range
                        && ranges.is_some_and(|ranges| {
                            origin.is_none_or(|origin| {
                                !ranges.entity_is_in_range_of_view(entity, origin)
                            })
                        }))
                {
                    return;
                }
                for (index, job) in jobs.iter().enumerate() {
                    // Range/inherited admission was evaluated once for all local lights.
                    let faces = local_caster_faces(
                        entity,
                        inherited,
                        mask,
                        aabb,
                        transform,
                        false,
                        no_frustum,
                        &job.layers,
                        &job.sphere,
                        &job.frusta,
                        origin,
                        ranges,
                    );
                    if faces != 0 {
                        thread[index].push((entity, faces));
                    }
                }
            },
        );
        for (index, job) in jobs.iter().enumerate() {
            cache.scratch.decisions.clear();
            for thread in queues.iter_mut() {
                cache.scratch.decisions.append(&mut thread[index]);
            }
            cache
                .scratch
                .decisions
                .sort_unstable_by_key(|(entity, _)| *entity);
            if job.point {
                let Ok((_, _, _, mut lists, _)) = point_lights.get_mut(job.entity) else {
                    continue;
                };
                let view = cache.points.entry(job.entity).or_default();
                if publish_full_local_shadow_view(
                    view,
                    job,
                    origin,
                    &cache.scratch.decisions,
                    lists.bypass_change_detection().iter_mut(),
                ) {
                    lists.set_changed();
                }
                publication.push(Arc::clone(&view.union));
            } else {
                let Ok((_, _, _, mut list, _)) = spot_lights.get_mut(job.entity) else {
                    continue;
                };
                let view = cache.spots.entry(job.entity).or_default();
                if publish_full_local_shadow_view(
                    view,
                    job,
                    origin,
                    &cache.scratch.decisions,
                    core::iter::once(list.bypass_change_detection()),
                ) {
                    list.set_changed();
                }
                publication.push(Arc::clone(&view.union));
            }
        }
    }
    cache
        .points
        .retain(|entity, _| checked_lights.contains(entity) && point_lights.get(*entity).is_ok());
    cache
        .spots
        .retain(|entity, _| checked_lights.contains(entity) && spot_lights.get(*entity).is_ok());

    if !publication.is_empty() {
        commands.queue(move |world: &mut World| {
            let mut query = world.query::<&mut ViewVisibility>();
            for entities in publication {
                let mut iter = query.iter_many_mut(world, entities.iter()).matched();
                while let Some(mut visibility) = iter.fetch_next() {
                    visibility.set_visible();
                }
            }
        });
    }
}

/// Determines the LOD origin for spot and point light shadow maps.
///
/// The selection priority is, from highest to lowest:
///
/// 1. An entity explicitly marked with the [`ShadowLodOrigin`] component.
///
/// 2. A camera that renders to a window.
///
/// 3. Any camera.
pub fn get_shadow_lod_origin(
    mut camera_query: QueryLens<(Entity, &RenderTarget), With<Camera>>,
    mut shadow_lod_origin_query: QueryLens<Entity, With<ShadowLodOrigin>>,
    mut lights_query: QueryLens<Entity, Or<(With<PointLight>, With<SpotLight>)>>,
) -> Option<Entity> {
    let (camera_query, shadow_lod_origin_query) =
        (camera_query.query(), shadow_lod_origin_query.query());

    let mut entities: SmallVec<[Entity; 4]> = smallvec![];
    entities.extend(shadow_lod_origin_query.iter());
    if let Some(lod_origin) = entities.iter().min() {
        return Some(*lod_origin);
    }

    entities.extend(
        camera_query
            .iter()
            .filter_map(|(main_entity, render_target)| match *render_target {
                RenderTarget::Window(_) => Some(main_entity),
                _ => None,
            }),
    );
    if let Some(lod_origin) = entities.iter().min() {
        return Some(*lod_origin);
    };

    entities.extend(camera_query.iter().map(|(main_entity, _)| main_entity));
    if let Some(lod_origin) = entities.iter().min() {
        if !lights_query.query().is_empty() {
            warn_once!(
                "Point lights and/or spot lights are present, but no entity has \
                 `ShadowLodOrigin`, and no camera that renders to the window has been found. \
                 Consider using the `ShadowLodOrigin` component to set a LOD origin."
            );
        }

        return Some(*lod_origin);
    };

    None
}

#[cfg(test)]
mod directional_shadow_receiver_tests {
    use super::*;
    use bevy_camera::PerspectiveProjection;
    use bevy_math::{Mat4, Vec3, Vec3A};
    use bevy_transform::components::Transform;

    #[test]
    fn incremental_camera_visibility_reapplies_union_and_reconciles_sources() {
        use bevy_camera::visibility::{check_visibility_cpu_culling, VisibilityClass};
        let mut app = App::new();
        app.add_plugins(bevy_app::TaskPoolPlugin::default());
        let world = app.world_mut();
        let view = world
            .spawn((
                Camera::default(),
                Frustum::default(),
                VisibleEntities::default(),
            ))
            .id();
        let mut entities = Vec::new();
        let class = TypeId::of::<Mesh3d>();
        for _ in 0..16 {
            entities.push(
                world
                    .spawn((
                        InheritedVisibility::VISIBLE,
                        ViewVisibility::HIDDEN,
                        GlobalTransform::IDENTITY,
                        NoFrustumCulling,
                        VisibilityClass(smallvec![class]),
                    ))
                    .id(),
            );
        }
        let mut schedule = Schedule::default();
        schedule.add_systems(check_visibility_cpu_culling);
        schedule.run(world);
        let initial = world
            .get::<VisibleEntities>(view)
            .unwrap()
            .get(class)
            .to_vec();
        assert_eq!(initial.len(), entities.len());
        let initial_tick = world
            .entity(view)
            .get_ref::<VisibleEntities>()
            .unwrap()
            .last_changed();
        for &entity in &entities {
            *world
                .get_mut::<ViewVisibility>(entity)
                .unwrap()
                .bypass_change_detection() = ViewVisibility::HIDDEN;
        }
        schedule.run(world);
        assert_eq!(
            world
                .entity(view)
                .get_ref::<VisibleEntities>()
                .unwrap()
                .last_changed(),
            initial_tick
        );
        assert!(entities
            .iter()
            .all(|&entity| world.get::<ViewVisibility>(entity).unwrap().get()));

        let first = entities[0];
        world.entity_mut(first).insert(RenderLayers::layer(1));
        *world
            .get_mut::<ViewVisibility>(first)
            .unwrap()
            .bypass_change_detection() = ViewVisibility::HIDDEN;
        schedule.run(world);
        assert!(!world
            .get::<VisibleEntities>(view)
            .unwrap()
            .get(class)
            .contains(&first));
        assert!(!world.get::<ViewVisibility>(first).unwrap().get());
        world.entity_mut(first).remove::<RenderLayers>();
        schedule.run(world);
        assert_eq!(
            world.get::<VisibleEntities>(view).unwrap().get(class),
            initial
        );
        world
            .entity_mut(first)
            .insert(VisibilityClass(smallvec![class, class]));
        schedule.run(world);
        assert_eq!(
            world
                .get::<VisibleEntities>(view)
                .unwrap()
                .get(class)
                .iter()
                .filter(|&&entity| entity == first)
                .count(),
            2
        );
        world.entity_mut(first).insert(NoCpuCulling);
        schedule.run(world);
        assert!(!world
            .get::<VisibleEntities>(view)
            .unwrap()
            .get(class)
            .contains(&first));
        world.entity_mut(first).remove::<NoCpuCulling>();
        schedule.run(world);
        assert_eq!(
            world
                .get::<VisibleEntities>(view)
                .unwrap()
                .get(class)
                .iter()
                .filter(|&&entity| entity == first)
                .count(),
            2
        );
        world.despawn(first);
        schedule.run(world);
        assert!(!world
            .get::<VisibleEntities>(view)
            .unwrap()
            .get(class)
            .contains(&first));
        world.entity_mut(view).insert(RenderLayers::layer(1));
        schedule.run(world);
        assert!(world
            .get::<VisibleEntities>(view)
            .unwrap()
            .get(class)
            .is_empty());
    }

    #[test]
    fn incremental_local_visibility_republishes_point_faces_and_spot_membership() {
        let mut app = App::new();
        app.add_plugins(bevy_app::TaskPoolPlugin::default());
        let world = app.world_mut();
        let spot = world
            .spawn((
                SpotLight {
                    shadow_maps_enabled: true,
                    ..Default::default()
                },
                Frustum::default(),
                VisibleMeshEntities::default(),
                GlobalTransform::IDENTITY,
            ))
            .id();
        let point = world
            .spawn((
                PointLight {
                    shadow_maps_enabled: true,
                    ..Default::default()
                },
                CubemapFrusta::default(),
                CubemapVisibleEntities::default(),
                GlobalTransform::IDENTITY,
            ))
            .id();
        let mut visible = VisibleEntities::default();
        visible.push(spot, TypeId::of::<ClusterVisibilityClass>());
        visible.push(point, TypeId::of::<ClusterVisibilityClass>());
        let camera = world.spawn(visible).id();
        let entities: Vec<_> = (0..16)
            .map(|_| {
                world
                    .spawn((
                        Mesh3d(Default::default()),
                        InheritedVisibility::VISIBLE,
                        ViewVisibility::HIDDEN,
                        NoFrustumCulling,
                        GlobalTransform::IDENTITY,
                    ))
                    .id()
            })
            .collect();
        let mut schedule = Schedule::default();
        schedule.add_systems(check_point_light_mesh_visibility);
        schedule.run(world);
        let spot_list = |world: &World| {
            world
                .get::<VisibleMeshEntities>(spot)
                .unwrap()
                .entities
                .clone()
        };
        let point_list = |world: &World| {
            world
                .get::<CubemapVisibleEntities>(point)
                .unwrap()
                .iter()
                .map(|face| face.entities.clone())
                .collect::<Vec<_>>()
        };
        let initial = spot_list(world);
        assert_eq!(initial.len(), entities.len());
        assert!(point_list(world).iter().all(|face| face == &initial));
        let tick = world
            .entity(spot)
            .get_ref::<VisibleMeshEntities>()
            .unwrap()
            .last_changed();
        for &entity in &entities {
            *world
                .get_mut::<ViewVisibility>(entity)
                .unwrap()
                .bypass_change_detection() = ViewVisibility::HIDDEN;
        }
        schedule.run(world);
        assert_eq!(
            world
                .entity(spot)
                .get_ref::<VisibleMeshEntities>()
                .unwrap()
                .last_changed(),
            tick
        );
        assert!(entities
            .iter()
            .all(|&e| world.get::<ViewVisibility>(e).unwrap().get()));

        world
            .entity_mut(spot)
            .insert(GlobalTransform::from_translation(Vec3::X));
        world
            .entity_mut(point)
            .insert(GlobalTransform::from_translation(Vec3::Z));
        schedule.run(world);
        assert_eq!(spot_list(world), initial);
        assert!(point_list(world).iter().all(|face| face == &initial));
        assert_eq!(
            world
                .entity(spot)
                .get_ref::<VisibleMeshEntities>()
                .unwrap()
                .last_changed(),
            tick,
        );
        let first = entities[0];
        world
            .entity_mut(first)
            .insert(bevy_ecs::entity_disabling::Disabled);
        schedule.run(world);
        assert!(!spot_list(world).contains(&first));
        assert!(point_list(world).iter().all(|face| !face.contains(&first)));
        world
            .entity_mut(first)
            .remove::<bevy_ecs::entity_disabling::Disabled>();
        schedule.run(world);
        assert_eq!(spot_list(world), initial);
        let tick = world
            .entity(spot)
            .get_ref::<VisibleMeshEntities>()
            .unwrap()
            .last_changed();
        world
            .entity_mut(first)
            .insert(GlobalTransform::from_translation(Vec3::X));
        schedule.run(world);
        assert_eq!(
            world
                .entity(spot)
                .get_ref::<VisibleMeshEntities>()
                .unwrap()
                .last_changed(),
            tick
        );
        world.entity_mut(first).insert(NotShadowCaster);
        *world
            .get_mut::<ViewVisibility>(first)
            .unwrap()
            .bypass_change_detection() = ViewVisibility::HIDDEN;
        schedule.run(world);
        assert!(!spot_list(world).contains(&first));
        assert!(point_list(world).iter().all(|face| !face.contains(&first)));
        assert!(!world.get::<ViewVisibility>(first).unwrap().get());
        world.entity_mut(first).remove::<NotShadowCaster>();
        schedule.run(world);
        assert_eq!(spot_list(world), initial);
        world.entity_mut(first).insert(RenderLayers::layer(1));
        schedule.run(world);
        assert!(!spot_list(world).contains(&first));
        assert!(point_list(world).iter().all(|face| !face.contains(&first)));
        world.entity_mut(first).remove::<RenderLayers>();
        schedule.run(world);
        assert_eq!(spot_list(world), initial);
        world.despawn(first);
        schedule.run(world);
        assert!(!spot_list(world).contains(&first));

        // An inactive light cannot reuse membership after changes while absent.
        world
            .get_mut::<VisibleEntities>(camera)
            .unwrap()
            .clear(TypeId::of::<ClusterVisibilityClass>());
        schedule.run(world);
        let second = entities[1];
        world.entity_mut(second).insert(NotShadowCaster);
        schedule.run(world);
        world
            .get_mut::<VisibleEntities>(camera)
            .unwrap()
            .push(spot, TypeId::of::<ClusterVisibilityClass>());
        schedule.run(world);
        assert!(!spot_list(world).contains(&second));
        assert_eq!(spot_list(world).len(), entities.len() - 2);
        world
            .get_mut::<SpotLight>(spot)
            .unwrap()
            .shadow_maps_enabled = false;
        schedule.run(world);
        world.entity_mut(second).remove::<NotShadowCaster>();
        world
            .get_mut::<SpotLight>(spot)
            .unwrap()
            .shadow_maps_enabled = true;
        schedule.run(world);
        assert_eq!(spot_list(world).len(), entities.len() - 1);
    }

    #[test]
    fn incremental_directional_visibility_reapplies_union_and_reconciles_sources() {
        let mut app = App::new();
        app.add_plugins(bevy_app::TaskPoolPlugin::default());
        let world = app.world_mut();
        let view = world.spawn_empty().id();
        let light = world
            .spawn((
                DirectionalLight {
                    shadow_maps_enabled: true,
                    ..Default::default()
                },
                ViewVisibility::VISIBLE,
                CascadesFrusta {
                    frusta: [(view, vec![Frustum::default(), Frustum::default()])]
                        .into_iter()
                        .collect(),
                },
            ))
            .id();
        let entities: Vec<_> = (0..16)
            .map(|_| {
                world
                    .spawn((
                        Mesh3d(Default::default()),
                        InheritedVisibility::VISIBLE,
                        ViewVisibility::HIDDEN,
                        NoFrustumCulling,
                        GlobalTransform::IDENTITY,
                    ))
                    .id()
            })
            .collect();
        let mut schedule = Schedule::default();
        schedule.add_systems(check_dir_light_mesh_visibility);
        schedule.run(world);
        let list = |world: &World| {
            world
                .get::<CascadesVisibleEntities>(light)
                .unwrap()
                .entities[&view][0]
                .entities
                .clone()
        };
        let initial = list(world);
        assert_eq!(initial.len(), entities.len());
        let initial_tick = world
            .entity(light)
            .get_ref::<CascadesVisibleEntities>()
            .unwrap()
            .last_changed();
        for &entity in &entities {
            *world
                .get_mut::<ViewVisibility>(entity)
                .unwrap()
                .bypass_change_detection() = ViewVisibility::HIDDEN;
        }
        schedule.run(world);
        assert_eq!(
            world
                .entity(light)
                .get_ref::<CascadesVisibleEntities>()
                .unwrap()
                .last_changed(),
            initial_tick
        );
        assert!(entities
            .iter()
            .all(|&entity| world.get::<ViewVisibility>(entity).unwrap().get()));
        let first = entities[0];
        world.entity_mut(first).insert(NotShadowCaster);
        *world
            .get_mut::<ViewVisibility>(first)
            .unwrap()
            .bypass_change_detection() = ViewVisibility::HIDDEN;
        schedule.run(world);
        assert!(!list(world).contains(&first));
        assert!(!world.get::<ViewVisibility>(first).unwrap().get());
        world.entity_mut(first).remove::<NotShadowCaster>();
        schedule.run(world);
        assert_eq!(list(world), initial);
        world.entity_mut(first).insert(RenderLayers::layer(1));
        schedule.run(world);
        assert!(!list(world).contains(&first));
        world.entity_mut(first).remove::<RenderLayers>();
        schedule.run(world);
        assert_eq!(list(world), initial);
        world.despawn(first);
        schedule.run(world);
        assert!(!list(world).contains(&first));
        world
            .get_mut::<DirectionalLight>(light)
            .unwrap()
            .shadow_maps_enabled = false;
        schedule.run(world);
        assert!(world
            .get::<CascadesVisibleEntities>(light)
            .unwrap()
            .entities
            .is_empty());
        world
            .get_mut::<DirectionalLight>(light)
            .unwrap()
            .shadow_maps_enabled = true;
        schedule.run(world);
        assert_eq!(list(world).len(), entities.len() - 1);
        world
            .get_mut::<CascadesFrusta>(light)
            .unwrap()
            .frusta
            .get_mut(&view)
            .unwrap()
            .pop();
        schedule.run(world);
        assert_eq!(
            world
                .get::<CascadesVisibleEntities>(light)
                .unwrap()
                .entities[&view]
                .len(),
            1
        );
    }

    fn receiver_bounds(min: DVec2, max: DVec2) -> DirectionalShadowReceiverBounds {
        receiver_hull(
            &[min, DVec2::new(max.x, min.y), max, DVec2::new(min.x, max.y)],
            DVec2::ZERO,
        )
    }

    fn receiver_hull(points: &[DVec2], padding: DVec2) -> DirectionalShadowReceiverBounds {
        let mut projected = SmallVec::<[DVec2; 8]>::new();
        projected.extend(points.iter().copied());
        let (vertices, vertex_count) = directional_shadow_receiver_hull(projected).unwrap();
        let hull = &vertices[..usize::from(vertex_count)];
        let axes = directional_shadow_receiver_axes(hull).unwrap();
        let min = hull
            .iter()
            .fold(DVec2::splat(f64::INFINITY), |min, point| min.min(*point))
            - padding;
        let max = hull
            .iter()
            .fold(DVec2::splat(f64::NEG_INFINITY), |max, point| {
                max.max(*point)
            })
            + padding;
        DirectionalShadowReceiverBounds {
            clip_from_world: DMat4::IDENTITY,
            axes,
            vertex_count,
            min,
            max,
            padding,
        }
    }

    #[test]
    fn rejects_a_caster_whose_projected_bounds_miss_the_receiver_slice() {
        let receivers = receiver_bounds(DVec2::splat(-1.0), DVec2::splat(1.0));
        let caster = Aabb::from_min_max(Vec3::new(2.0, -0.25, -100.0), Vec3::new(3.0, 0.25, 100.0));

        assert!(aabb_is_outside_directional_shadow_receiver_bounds(
            &caster,
            &GlobalTransform::IDENTITY,
            &receivers,
        ));
    }

    #[test]
    fn retains_an_upstream_caster_that_can_shadow_a_receiver() {
        let receivers = receiver_bounds(DVec2::splat(-1.0), DVec2::splat(1.0));
        let caster = Aabb::from_min_max(
            Vec3::new(-0.25, -0.25, 100_000.0),
            Vec3::new(0.25, 0.25, 100_010.0),
        );

        assert!(!aabb_is_outside_directional_shadow_receiver_bounds(
            &caster,
            &GlobalTransform::IDENTITY,
            &receivers,
        ));
    }

    #[test]
    fn rejects_a_caster_inside_the_receiver_rectangle_but_outside_its_hull() {
        let receivers = receiver_hull(
            &[
                DVec2::new(0.0, -1.0),
                DVec2::new(1.0, 0.0),
                DVec2::new(0.0, 1.0),
                DVec2::new(-1.0, 0.0),
            ],
            DVec2::ZERO,
        );
        let caster = Aabb::from_min_max(Vec3::new(0.75, 0.75, -10.0), Vec3::new(0.95, 0.95, 10.0));

        assert!(aabb_is_outside_directional_shadow_receiver_bounds(
            &caster,
            &GlobalTransform::IDENTITY,
            &receivers,
        ));
    }

    #[test]
    fn retains_a_caster_intersecting_the_receiver_hull() {
        let receivers = receiver_hull(
            &[
                DVec2::new(0.0, -1.0),
                DVec2::new(1.0, 0.0),
                DVec2::new(0.0, 1.0),
                DVec2::new(-1.0, 0.0),
            ],
            DVec2::ZERO,
        );
        let caster = Aabb::from_min_max(
            Vec3::new(0.35, 0.35, 100_000.0),
            Vec3::new(0.55, 0.55, 100_010.0),
        );

        assert!(!aabb_is_outside_directional_shadow_receiver_bounds(
            &caster,
            &GlobalTransform::IDENTITY,
            &receivers,
        ));
    }

    #[test]
    fn receiver_hull_padding_preserves_shadow_filter_footprints() {
        let receivers = receiver_hull(
            &[
                DVec2::new(0.0, -1.0),
                DVec2::new(1.0, 0.0),
                DVec2::new(0.0, 1.0),
                DVec2::new(-1.0, 0.0),
            ],
            DVec2::splat(0.1),
        );
        let caster = Aabb::from_min_max(Vec3::new(0.55, 0.55, -10.0), Vec3::new(0.65, 0.65, 10.0));

        assert!(!aabb_is_outside_directional_shadow_receiver_bounds(
            &caster,
            &GlobalTransform::IDENTITY,
            &receivers,
        ));
    }

    #[test]
    fn invalid_caster_bounds_remain_visible_to_the_native_culling_path() {
        let receivers = receiver_bounds(DVec2::splat(-1.0), DVec2::splat(1.0));
        let caster = Aabb {
            center: Vec3A::splat(f32::NAN),
            half_extents: Vec3A::ONE,
        };

        assert!(!aabb_is_outside_directional_shadow_receiver_bounds(
            &caster,
            &GlobalTransform::IDENTITY,
            &receivers,
        ));
    }

    #[test]
    fn degenerate_projected_hull_uses_native_culling() {
        let mut points = SmallVec::<[DVec2; 8]>::new();
        points.extend([
            DVec2::new(0.0, 0.0),
            DVec2::new(1.0, 1.0),
            DVec2::new(2.0, 2.0),
        ]);

        assert!(directional_shadow_receiver_hull(points).is_none());
    }

    #[test]
    fn non_finite_projected_hull_uses_native_culling() {
        let mut points = SmallVec::<[DVec2; 8]>::new();
        points.extend([
            DVec2::new(0.0, 0.0),
            DVec2::new(1.0, 0.0),
            DVec2::new(0.0, f64::NAN),
        ]);

        assert!(directional_shadow_receiver_hull(points).is_none());
    }

    #[test]
    fn invalid_camera_transform_uses_native_shadow_culling() {
        let projection = Projection::Perspective(PerspectiveProjection::default());
        let world_from_view =
            GlobalTransform::from(Transform::from_translation(Vec3::splat(f32::NAN)));
        let mut cascade = Cascade::default();
        cascade.clip_from_world = Mat4::IDENTITY;
        cascade.texel_size = 0.1;

        assert!(directional_shadow_receiver_bounds(
            &projection,
            &world_from_view,
            &cascade,
            1.0,
            10.0,
            UVec2::new(2000, 1000),
            1.0,
            2.0,
        )
        .is_none());
    }

    #[test]
    fn receiver_bounds_include_filter_and_normal_bias_footprints() {
        let projection = Projection::Perspective(PerspectiveProjection::default());
        let mut cascade = Cascade::default();
        cascade.clip_from_world = Mat4::IDENTITY;
        cascade.texel_size = 0.1;
        let corners = projection.get_frustum_corners(-1.0, -10.0);
        let raw_min_x = corners
            .iter()
            .map(|corner| f64::from(corner.x))
            .fold(f64::INFINITY, f64::min);
        let raw_max_x = corners
            .iter()
            .map(|corner| f64::from(corner.x))
            .fold(f64::NEG_INFINITY, f64::max);

        let bounds = directional_shadow_receiver_bounds(
            &projection,
            &GlobalTransform::IDENTITY,
            &cascade,
            1.0,
            10.0,
            UVec2::new(2000, 1000),
            1.0,
            DirectionalLight {
                shadow_normal_bias: 2.0,
                ..Default::default()
            }
            .receiver_normal_bias(),
        )
        .unwrap();

        assert!(bounds.min.x <= raw_min_x - 0.38);
        assert!(bounds.max.x >= raw_max_x + 0.38);
    }

    #[test]
    fn frustum_receiver_bounds_cover_half_pixel_temporal_jitter() {
        let projection = Projection::Perspective(PerspectiveProjection {
            fov: core::f32::consts::FRAC_PI_2,
            aspect_ratio: 2.0,
            ..Default::default()
        });
        let viewport_size = UVec2::new(2000, 1000);
        let original = projection.get_frustum_corners(-1.0, -10.0);
        let expanded =
            frustum_corners_with_temporal_jitter_margin(&projection, 1.0, 10.0, viewport_size)
                .unwrap();
        let far_width = original[4].x - original[6].x;
        let expected_x_margin = far_width / viewport_size.x as f32 * 0.5;
        let far_height = original[5].y - original[4].y;
        let expected_y_margin = far_height / viewport_size.y as f32 * 0.5;

        assert!(
            (expanded[4].x - original[4].x - expected_x_margin).abs() < 2e-5,
            "original={} expanded={} expected_margin={}",
            original[4].x,
            expanded[4].x,
            expected_x_margin,
        );
        assert!((original[6].x - expanded[6].x - expected_x_margin).abs() < 2e-5);
        assert!((original[4].y - expanded[4].y - expected_y_margin).abs() < 2e-5);
        assert!((expanded[5].y - original[5].y - expected_y_margin).abs() < 2e-5);
    }

    #[test]
    fn temporal_filter_uses_native_culling_without_an_extra_bound() {
        assert_eq!(
            directional_shadow_filter_radius(ShadowFilteringMethod::Hardware2x2),
            Some(1.0),
        );
        assert_eq!(
            directional_shadow_filter_radius(ShadowFilteringMethod::Gaussian),
            Some(3.0),
        );
        assert_eq!(
            directional_shadow_filter_radius(ShadowFilteringMethod::Temporal),
            None,
        );
    }

    #[test]
    fn directional_filter_support_matches_the_available_shadow_footprint() {
        let mut light = DirectionalLight::default();
        assert!(directional_shadow_receiver_culling_supported(&light));

        #[cfg(feature = "experimental_pbr_pcss")]
        {
            light.soft_shadow_size = Some(1.0);
            assert!(!directional_shadow_receiver_culling_supported(&light));
        }
    }
}

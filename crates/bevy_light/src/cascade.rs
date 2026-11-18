//! Provides shadow cascade configuration and construction helpers.

use bevy_camera::{Camera, Projection};
use bevy_ecs::{entity::EntityHashMap, prelude::*};
use bevy_math::{ops, DVec2, DVec3, Mat4, Quat, Vec3, Vec3A, Vec4};
use bevy_platform::collections::HashMap;
use bevy_reflect::prelude::*;
use bevy_transform::components::GlobalTransform;

use crate::{DirectionalLight, DirectionalLightShadowMap};

/// Controls how cascaded shadow mapping works.
/// Prefer using [`CascadeShadowConfigBuilder`] to construct an instance.
///
/// ```
/// # use bevy_light::CascadeShadowConfig;
/// # use bevy_light::CascadeShadowConfigBuilder;
/// # use bevy_utils::default;
/// #
/// let config: CascadeShadowConfig = CascadeShadowConfigBuilder {
///   maximum_distance: 100.0,
///   ..default()
/// }.into();
/// ```
#[derive(Component, Clone, Debug, Reflect)]
#[reflect(Component, Default, Debug, Clone)]
pub struct CascadeShadowConfig {
    /// The (positive) distance to the far boundary of each cascade.
    pub bounds: Vec<f32>,
    /// The proportion of overlap each cascade has with the previous cascade.
    pub overlap_proportion: f32,
    /// The (positive) distance to the near boundary of the first cascade.
    pub minimum_distance: f32,
}

impl Default for CascadeShadowConfig {
    fn default() -> Self {
        CascadeShadowConfigBuilder::default().into()
    }
}

fn calculate_cascade_bounds(
    num_cascades: usize,
    nearest_bound: f32,
    shadow_maximum_distance: f32,
) -> Vec<f32> {
    if num_cascades == 1 {
        return vec![shadow_maximum_distance];
    }
    let base = ops::powf(
        shadow_maximum_distance / nearest_bound,
        1.0 / (num_cascades - 1) as f32,
    );
    (0..num_cascades)
        .map(|i| nearest_bound * ops::powf(base, i as f32))
        .collect()
}

/// Builder for [`CascadeShadowConfig`].
#[derive(Clone)]
pub struct CascadeShadowConfigBuilder {
    /// The number of shadow cascades.
    /// More cascades increases shadow quality by mitigating perspective aliasing - a phenomenon where areas
    /// nearer the camera are covered by fewer shadow map texels than areas further from the camera, causing
    /// blocky looking shadows.
    ///
    /// This does come at the cost increased rendering overhead, however this overhead is still less
    /// than if you were to use fewer cascades and much larger shadow map textures to achieve the
    /// same quality level.
    ///
    /// In case rendered geometry covers a relatively narrow and static depth relative to camera, it may
    /// make more sense to use fewer cascades and a higher resolution shadow map texture as perspective aliasing
    /// is not as much an issue. Be sure to adjust `minimum_distance` and `maximum_distance` appropriately.
    pub num_cascades: usize,
    /// The minimum shadow distance, which can help improve the texel resolution of the first cascade.
    /// Areas nearer to the camera than this will likely receive no shadows.
    ///
    /// NOTE: Due to implementation details, this usually does not impact shadow quality as much as
    /// `first_cascade_far_bound` and `maximum_distance`. At many view frustum field-of-views, the
    /// texel resolution of the first cascade is dominated by the width / height of the view frustum plane
    /// at `first_cascade_far_bound` rather than the depth of the frustum from `minimum_distance` to
    /// `first_cascade_far_bound`.
    pub minimum_distance: f32,
    /// The maximum shadow distance.
    /// Areas further from the camera than this will likely receive no shadows.
    pub maximum_distance: f32,
    /// Sets the far bound of the first cascade, relative to the view origin.
    /// In-between cascades will be exponentially spaced relative to the maximum shadow distance.
    /// NOTE: This is ignored if there is only one cascade, the maximum distance takes precedence.
    pub first_cascade_far_bound: f32,
    /// Sets the overlap proportion between cascades.
    /// The overlap is used to make the transition from one cascade's shadow map to the next
    /// less abrupt by blending between both shadow maps.
    pub overlap_proportion: f32,
}

impl CascadeShadowConfigBuilder {
    /// Returns the cascade config as specified by this builder.
    pub fn build(&self) -> CascadeShadowConfig {
        assert!(
            self.num_cascades > 0,
            "num_cascades must be positive, but was {}",
            self.num_cascades
        );
        assert!(
            self.minimum_distance >= 0.0,
            "maximum_distance must be non-negative, but was {}",
            self.minimum_distance
        );
        assert!(
            self.num_cascades == 1 || self.minimum_distance < self.first_cascade_far_bound,
            "minimum_distance must be less than first_cascade_far_bound, but was {}",
            self.minimum_distance
        );
        assert!(
            self.maximum_distance > self.minimum_distance,
            "maximum_distance must be greater than minimum_distance, but was {}",
            self.maximum_distance
        );
        assert!(
            (0.0..1.0).contains(&self.overlap_proportion),
            "overlap_proportion must be in [0.0, 1.0) but was {}",
            self.overlap_proportion
        );
        CascadeShadowConfig {
            bounds: calculate_cascade_bounds(
                self.num_cascades,
                self.first_cascade_far_bound,
                self.maximum_distance,
            ),
            overlap_proportion: self.overlap_proportion,
            minimum_distance: self.minimum_distance,
        }
    }
}

impl Default for CascadeShadowConfigBuilder {
    fn default() -> Self {
        // The defaults are chosen to be similar to be Unity, Unreal, and Godot.
        // Unity: first cascade far bound = 10.05, maximum distance = 150.0
        // Unreal Engine 5: maximum distance = 200.0
        // Godot: first cascade far bound = 10.0, maximum distance = 100.0
        Self {
            // Currently only support one cascade in WebGL 2.
            num_cascades: if cfg!(all(
                feature = "webgl",
                target_arch = "wasm32",
                not(feature = "webgpu")
            )) {
                1
            } else {
                4
            },
            minimum_distance: 0.1,
            maximum_distance: 150.0,
            first_cascade_far_bound: 10.0,
            overlap_proportion: 0.2,
        }
    }
}

impl From<CascadeShadowConfigBuilder> for CascadeShadowConfig {
    fn from(builder: CascadeShadowConfigBuilder) -> Self {
        builder.build()
    }
}

/// On a camera, identifies a rigid receiving frame for directional shadow sampling.
/// Its origin anchors the texel lattice and its rotation carries the transverse
/// axes across render-world rebases. The caster and camera transforms must be
/// expressed in the same finalized render world as this entity.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct DirectionalShadowReference {
    /// Identity of the rigid coordinate system, including its entity generation.
    pub frame: Entity,
    /// Orientation of the reference axes in the finalized render world.
    pub rotation: Quat,
    /// A nearby fixed point, expressed in the finalized render world.
    pub render_origin: Vec3,
    /// The same point in the persistent coordinate system. Moving to an adjacent
    /// reference patch carries the shadow grid phase across the boundary.
    pub frame_origin: DVec3,
}

/// Persistent axes in receiving-frame coordinates, shared across cameras at a site.
#[derive(Default)]
pub struct ShadowFrameBases {
    frames: HashMap<(Entity, Entity), ShadowFrame>,
}

struct ShadowFrame {
    datum: DVec3,
    right: Vec3,
    direction: Vec3,
}

fn transported_right(previous_x: Vec3, previous_z: Vec3, z: Vec3) -> Vec3 {
    let denominator = 1.0 + previous_z.dot(z);
    // At the antipode both half turns are valid. Preserve the previous X axis.
    let x = if denominator > 1.0e-5 {
        previous_x - (previous_z + z) * (previous_x.dot(z) / denominator)
    } else {
        previous_x
    };
    (x - z * x.dot(z)).normalize()
}

impl ShadowFrameBases {
    fn projection_frame(
        &mut self,
        light: Entity,
        reference: DirectionalShadowReference,
        world_z: Vec3,
    ) -> (Mat4, DVec2) {
        let frame = reference.rotation;
        let z = (frame.inverse() * world_z).normalize();
        let state = self
            .frames
            .entry((light, reference.frame))
            .or_insert_with(|| ShadowFrame {
                datum: reference.frame_origin,
                right: z.any_orthonormal_vector(),
                direction: z,
            });
        let x = transported_right(state.right, state.direction, z);
        state.right = x;
        state.direction = z;
        let y = z.cross(x);
        let delta = reference.frame_origin - state.datum;
        let offset = DVec2::new(delta.dot(x.as_dvec3()), delta.dot(y.as_dvec3()));
        (
            Mat4::from_cols(
                (frame * x).extend(0.0),
                (frame * y).extend(0.0),
                world_z.extend(0.0),
                Vec4::W,
            ),
            offset,
        )
    }
}

/// A [`DirectionalLight`]'s per-view list of [`Cascade`]s.
#[derive(Component, Clone, Debug, Default, Reflect)]
#[reflect(Component, Debug, Default, Clone)]
pub struct Cascades {
    /// Map from a view to the configuration of each of its [`Cascade`]s.
    pub cascades: EntityHashMap<Vec<Cascade>>,
}

/// A single cascade of a view's shadow map cascade. Several of these are
/// used to cover most of the view to ensure most geometry gets shadows, with
/// some overlap for blending at cascade transitions. Farther away cascades
/// are larger and have a lower effective shadowmap texel per world unit
/// resolution. All cascades have the same pixel dimensions however.
#[derive(Clone, Debug, Default, Reflect)]
#[reflect(Clone, Default)]
pub struct Cascade {
    /// The transform of the light, i.e. the view to world matrix.
    pub world_from_cascade: Mat4,
    /// The orthographic projection for this cascade.
    pub clip_from_cascade: Mat4,
    /// The view-projection matrix for this cascade, converting world space into light clip space.
    /// Importantly, this is derived and stored separately from `view_transform` and `projection` to
    /// ensure shadow stability.
    pub clip_from_world: Mat4,
    /// Size of each shadow map texel in world units.
    pub texel_size: f32,
}

/// Sets up [`Cascades`] for all shadow mapped [`DirectionalLight`]s.
pub fn build_directional_light_cascades(
    directional_light_shadow_map: Res<DirectionalLightShadowMap>,
    views: Query<(
        Entity,
        &GlobalTransform,
        &Projection,
        &Camera,
        Option<&DirectionalShadowReference>,
        Option<&crate::ShadowFilteringMethod>,
    )>,
    mut bases: Local<ShadowFrameBases>,
    entities: Query<Entity>,
    mut lights: Query<(
        Entity,
        &GlobalTransform,
        &DirectionalLight,
        &CascadeShadowConfig,
        &mut Cascades,
    )>,
) {
    let mut views = views
        .iter()
        .filter_map(
            |(entity, transform, projection, camera, reference, filter)| {
                if camera.is_active {
                    Some((
                        entity,
                        projection,
                        transform.to_matrix(),
                        reference.copied(),
                        filter.copied().unwrap_or_default(),
                    ))
                } else {
                    None
                }
            },
        )
        .collect::<Vec<_>>();

    views.sort_unstable_by_key(|view| view.0);
    bases
        .frames
        .retain(|(light, frame), _| lights.contains(*light) && entities.contains(*frame));

    for (light, transform, directional_light, cascades_config, mut cascades) in &mut lights {
        if !directional_light.shadow_maps_enabled {
            continue;
        }
        cascades.cascades.clear();

        // It is very important to the numerical and thus visual stability of shadows that
        // `world_from_light` has orthogonal upper-left 3x3 and zero translation.
        // Even though only the direction (i.e. rotation) of the light matters, we don't constrain
        // users to not change any other aspects of the transform - there's no guarantee
        // `transform.to_matrix()` will give us a matrix with our desired properties.
        // Instead, we directly create a good matrix from just the rotation.
        let native_basis = Mat4::from_quat(transform.rotation());
        for (view_entity, projection, world_from_view, reference, filter) in views.iter().copied() {
            let (world_from_light, offset) = match reference {
                Some(reference) => {
                    bases.projection_frame(light, reference, native_basis.z_axis.truncate())
                }
                None => (native_basis, DVec2::ZERO),
            };
            let anchor = reference.map(|r| r.render_origin);
            let light_from_world = world_from_light.transpose();
            let mut camera = world_from_view;
            if let Some(origin) = anchor {
                // Subtract the common render displacement before rotating or fitting.
                camera.w_axis = (camera.w_axis.truncate() - origin).extend(1.0);
            }
            let light_view_from_camera = light_from_world * camera;
            let overlap_factor = 1.0 - cascades_config.overlap_proportion;
            let far_bounds = cascades_config.bounds.iter();
            let near_bounds = [cascades_config.minimum_distance]
                .into_iter()
                .chain(far_bounds.clone().map(|bound| overlap_factor * bound));
            let view_cascades = near_bounds
                .zip(far_bounds)
                .map(|(near_bound, far_bound)| {
                    // Negate bounds as -z is camera forward direction.
                    let corners = projection.get_frustum_corners(-near_bound, -far_bound);
                    calculate_cascade(
                        corners,
                        directional_light_shadow_map.size as f32,
                        world_from_light,
                        light_view_from_camera,
                        anchor.map(|origin| (origin, offset)),
                        crate::directional_shadow_filter_radius(filter).unwrap_or(0.0) as f32,
                        directional_light,
                    )
                })
                .collect();
            cascades.cascades.insert(view_entity, view_cascades);
        }
    }
}

/// Returns a [`Cascade`] for the frustum defined by `frustum_corners`.
///
/// The corner vertices should be specified in the following order:
/// first the bottom right, top right, top left, bottom left for the near plane, then similar for the far plane.
///
/// See this [reference](https://developer.download.nvidia.com/SDK/10.5/opengl/src/cascaded_shadow_maps/doc/cascaded_shadow_maps.pdf) for more details.
fn calculate_cascade(
    frustum_corners: [Vec3A; 8],
    cascade_texture_size: f32,
    world_from_light: Mat4,
    light_from_camera: Mat4,
    anchor: Option<(Vec3, DVec2)>,
    filter_radius: f32,
    light: &DirectionalLight,
) -> Cascade {
    let mut min = Vec3A::splat(f32::MAX);
    let mut max = Vec3A::splat(f32::MIN);
    for corner_camera_view in frustum_corners {
        let corner_light_view = light_from_camera.transform_point3a(corner_camera_view);
        min = min.min(corner_light_view);
        max = max.max(corner_light_view);
    }

    // NOTE: Use the larger of the frustum slice far plane diagonal and body diagonal lengths as this
    //       will be the maximum possible projection size. Use the ceiling to get an integer which is
    //       very important for floating point stability later. It is also important that these are
    //       calculated using the original camera space corner positions for floating point precision
    //       as even though the lengths using corner_light_view above should be the same, precision can
    //       introduce small but significant differences.
    // NOTE: The size remains the same unless the view frustum or cascade configuration is modified.
    let body_diagonal = (frustum_corners[0] - frustum_corners[6]).length_squared();
    let far_plane_diagonal = (frustum_corners[4] - frustum_corners[6]).length_squared();
    let cascade_diameter = body_diagonal.max(far_plane_diagonal).sqrt().ceil();

    // Flooring the center spends up to one texel on one side. The receiver
    // offset and the filter must still fit after that translation.
    let normal_bias = light.receiver_normal_bias().abs();
    let border = 2.0 * (1.0 + filter_radius + normal_bias);
    let cascade_diameter = if cascade_texture_size > border {
        cascade_diameter * (cascade_texture_size / (cascade_texture_size - border))
    } else {
        cascade_diameter
    };
    let cascade_texel_size = cascade_diameter / cascade_texture_size;
    // NOTE: For shadow stability it is very important that the near_plane_center is at integer
    //       multiples of the texel size to be exactly representable in a floating point value.
    let center = 0.5 * (min + max);
    let snapped = if let Some((_, offset)) = anchor {
        // Only the fractional phase matters. Reduce it in double precision
        // before combining it with a bounded patch-relative center.
        let pitch = f64::from(cascade_texel_size);
        let phase = (offset % pitch).as_vec2();
        ((center.truncate() + phase) / cascade_texel_size).floor() * cascade_texel_size - phase
    } else {
        (center.truncate() / cascade_texel_size).floor() * cascade_texel_size
    };
    let depth_guard = normal_bias * cascade_texel_size + light.shadow_depth_bias.abs();
    let near_plane_center = Vec3A::new(snapped.x, snapped.y, max.z + depth_guard);

    // It is critical for `cascade_from_world` to be stable. So rather than forming `world_from_cascade`
    // and inverting it, which risks instability due to numerical precision, we directly form
    // `cascade_from_world` as the reference material suggests.
    let world_from_light_transpose = world_from_light.transpose();
    let local_center = Vec3::from(near_plane_center);
    let (light_center, world_center) = if let Some((origin, _)) = anchor {
        let basis = world_from_light.as_dmat4();
        let reference = origin.as_dvec3();
        let local = local_center.as_dvec3();
        // Convert once at the GPU matrix boundary. Construct both transforms
        // from the same center, without inverting a translated f32 matrix.
        (
            (local + basis.transpose().transform_vector3(reference)).as_vec3(),
            (reference + basis.transform_vector3(local)).as_vec3(),
        )
    } else {
        (
            local_center,
            world_from_light.transform_vector3(local_center),
        )
    };
    let cascade_from_world = Mat4::from_cols(
        world_from_light_transpose.x_axis,
        world_from_light_transpose.y_axis,
        world_from_light_transpose.z_axis,
        (-light_center).extend(1.0),
    );
    let world_from_cascade = Mat4::from_cols(
        world_from_light.x_axis,
        world_from_light.y_axis,
        world_from_light.z_axis,
        world_center.extend(1.0),
    );

    // Right-handed orthographic projection, centered at `near_plane_center`.
    // NOTE: This is different from the reference material, as we use reverse Z.
    let r = (max.z - min.z + 2.0 * depth_guard).recip();
    let clip_from_cascade = Mat4::from_cols(
        Vec4::new(2.0 / cascade_diameter, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 / cascade_diameter, 0.0, 0.0),
        Vec4::new(0.0, 0.0, r, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
    );

    let clip_from_world = clip_from_cascade * cascade_from_world;
    Cascade {
        world_from_cascade,
        clip_from_cascade,
        clip_from_world,
        texel_size: cascade_texel_size,
    }
}

#[cfg(test)]
mod reference_tests {
    use super::*;

    #[test]
    fn reference_grid_survives_rebase_roll_and_patch_handoff() {
        let light = Entity::from_bits(1);
        let frame = Entity::from_bits(3);
        let mut bases = ShadowFrameBases::default();
        let z = Vec3::new(0.11, 0.992, 0.05).normalize();
        let corners = [
            (2., -2., -1.),
            (2., 2., -1.),
            (-2., 2., -1.),
            (-2., -2., -1.),
            (2., -2., -9.),
            (2., 2., -9.),
            (-2., 2., -9.),
            (-2., -2., -9.),
        ]
        .map(|(x, y, z)| Vec3A::new(x, y, z));
        let camera = Vec3::new(32.2, 6.1, -13.2);
        let point = camera + Vec3::new(0.2, 0.8, -4.6);
        let mut phase = |rotation: Quat, shift: Vec3, origin: Vec3| {
            let reference = DirectionalShadowReference {
                frame,
                rotation,
                frame_origin: origin.as_dvec3(),
                render_origin: rotation * origin + shift,
            };
            let (basis, offset) = bases.projection_frame(light, reference, rotation * z);
            let camera_local =
                Mat4::from_rotation_translation(rotation, rotation * (camera - origin));
            let c = calculate_cascade(
                corners,
                4096.,
                basis,
                basis.transpose() * camera_local,
                Some((reference.render_origin, offset)),
                3.0,
                &DirectionalLight {
                    shadow_normal_bias: 2.0,
                    shadow_depth_bias: 0.005,
                    ..Default::default()
                },
            );
            for corner in corners {
                let world = rotation * (camera + Vec3::from(corner)) + shift;
                let uv = c.clip_from_world.transform_point3(world);
                let safe_edge = 1.0 - 2.0 * (3.0 + 2.0 * core::f32::consts::SQRT_2) / 4096.0;
                assert!(uv.x.abs() <= safe_edge && uv.y.abs() <= safe_edge);
                let biased = world + basis.z_axis.truncate() * (2.0 * c.texel_size + 0.005);
                assert!(c.clip_from_world.transform_point3(biased).z <= 1.0001);
                assert!((-0.0001..=1.0001).contains(&uv.z));
            }
            c.clip_from_world
                .transform_point3(rotation * point + shift)
                .truncate()
                * 2048.0
        };
        let a = phase(Quat::IDENTITY, Vec3::ZERO, Vec3::ZERO);
        let rot = Quat::from_rotation_y(0.7);
        let b = phase(rot, Vec3::new(327.7, 19.1, -518.3), Vec3::ZERO);
        let c = phase(
            rot,
            Vec3::new(327.7, 19.1, -518.3),
            Vec3::new(40., 0., -10.),
        );
        for uv in [b, c] {
            let difference = uv - a;
            assert!(
                (difference - difference.round()).abs().max_element() < 0.06,
                "{difference:?}"
            );
        }
        let mut prev = z;
        let mut x = z.any_orthonormal_vector();
        for angle in -150..150 {
            let next = Vec3::new(angle as f32 * 0.001, 1., 0.).normalize();
            let transported = transported_right(x, prev, next);
            assert!(transported.dot(x) > 0.95 && transported.dot(next).abs() < 1.0e-5);
            (x, prev) = (transported, next);
        }
        assert!(transported_right(x, prev, -prev).dot(x) > 0.999);
    }
}

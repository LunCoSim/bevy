//! A [mesh ray casting](ray_cast) backend for [`bevy_picking`](crate).
//!
//! By default, all meshes that have [`bevy_asset::RenderAssetUsages::MAIN_WORLD`] are pickable.
//! Picking can be disabled for individual entities by adding [`Pickable::IGNORE`].
//!
//! To make mesh picking entirely opt-in, set [`MeshPickingSettings::require_markers`]
//! to `true` and add [`MeshPickingCamera`] and [`Pickable`] components to the desired camera and
//! target entities.
//!
//! To manually perform mesh ray casts independent of picking, use the [`MeshRayCast`] system parameter.
//!
//! ## Implementation Notes
//!
//! - The `position` reported in `HitData` is in world space. The `normal` is a vector pointing
//!   away from the face, it is not guaranteed to be normalized for scaled meshes.

pub mod ray_cast;

use crate::{
    backend::{
        ray::{RayId, RayMap},
        HitData, PointerHits,
    },
    prelude::*,
    PickingSystems,
};
use bevy_app::prelude::*;
use bevy_asset::AssetId;
use bevy_camera::{primitives::Aabb, visibility::RenderLayers, Camera};
use bevy_ecs::prelude::*;
use bevy_mesh::Mesh;
use bevy_platform::collections::HashMap;
use bevy_reflect::prelude::*;
use bevy_shape::Ray3d;
use bevy_transform::components::GlobalTransform;
use ray_cast::{MeshRayCast, MeshRayCastSettings, RayCastVisibility};

/// An optional component that marks cameras that should be used in the [`MeshPickingPlugin`].
///
/// Only needed if [`MeshPickingSettings::require_markers`] is set to `true`, and ignored otherwise.
#[derive(Debug, Clone, Default, Component, Reflect)]
#[reflect(Debug, Default, Component)]
pub struct MeshPickingCamera;

/// Runtime settings for the [`MeshPickingPlugin`].
#[derive(Resource, Reflect)]
#[reflect(Resource, Default)]
pub struct MeshPickingSettings {
    /// When set to `true` ray casting will only consider cameras marked with
    /// [`MeshPickingCamera`] and entities marked with [`Pickable`]. `false` by default.
    ///
    /// This setting is provided to give you fine-grained control over which cameras and entities
    /// should be used by the mesh picking backend at runtime.
    pub require_markers: bool,

    /// Determines how mesh picking should consider [`Visibility`](bevy_camera::visibility::Visibility).
    /// When set to [`RayCastVisibility::Any`], ray casts can be performed against both visible and hidden entities.
    ///
    /// Defaults to [`RayCastVisibility::VisibleInView`], only performing picking against visible entities
    /// that are in the view of a camera.
    pub ray_cast_visibility: RayCastVisibility,
}

impl Default for MeshPickingSettings {
    fn default() -> Self {
        Self {
            require_markers: false,
            ray_cast_visibility: RayCastVisibility::VisibleInView,
        }
    }
}

/// Adds the mesh picking backend to your app.
#[derive(Clone, Default)]
pub struct MeshPickingPlugin;

impl Plugin for MeshPickingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MeshPickingSettings>()
            .add_systems(PreUpdate, update_hits.in_set(PickingSystems::Backend));
    }
}

/// Exact dependencies of one possible broad-phase candidate. Reading the
/// value rather than a change tick handles visibility's per-frame reset/write.
#[derive(PartialEq)]
struct CandidateSnapshot {
    entity: Entity,
    visible: bool,
    bounds: Aabb,
    transform: GlobalTransform,
    pickable: Option<Pickable>,
    layers: Option<RenderLayers>,
    mesh2d: Option<AssetId<Mesh>>,
    mesh3d: Option<AssetId<Mesh>>,
    simplified: Option<AssetId<Mesh>>,
    backfaces: bool,
}

#[derive(PartialEq)]
struct RaySnapshot {
    id: RayId,
    ray: Ray3d,
    order: f32,
    layers: RenderLayers,
}

struct CachedRay {
    source: RaySnapshot,
    picks: Vec<(Entity, HitData)>,
}

/// One previous complete candidate snapshot and the hits computed against it.
/// Motion of any admitted candidate invalidates reuse, even with a static cursor.
#[derive(Default)]
pub struct MeshPickingCache {
    rays: HashMap<RayId, CachedRay>,
    candidates: Vec<CandidateSnapshot>,
    scratch: Vec<CandidateSnapshot>,
    settings: Option<(bool, RayCastVisibility)>,
}

/// Casts rays into the scene using [`MeshPickingSettings`] and sends [`PointerHits`] events.
pub fn update_hits(
    backend_settings: Res<MeshPickingSettings>,
    ray_map: Res<RayMap>,
    picking_cameras: Query<(&Camera, Has<MeshPickingCamera>, Option<&RenderLayers>)>,
    pickables: Query<&Pickable>,
    marked_targets: Query<&Pickable>,
    layers: Query<&RenderLayers>,
    mut ray_cast: MeshRayCast,
    mut pointer_hits_writer: MessageWriter<PointerHits>,
    mut cache: Local<MeshPickingCache>,
) {
    let next_settings = (
        backend_settings.require_markers,
        backend_settings.ray_cast_visibility,
    );
    let mut rays = Vec::new();
    for (&ray_id, &ray) in ray_map.iter() {
        let Ok((camera, cam_can_pick, cam_layers)) = picking_cameras.get(ray_id.camera) else {
            continue;
        };
        if backend_settings.require_markers && !cam_can_pick {
            continue;
        }

        rays.push(RaySnapshot {
            id: ray_id,
            ray,
            order: camera.order as f32,
            layers: cam_layers.cloned().unwrap_or_default(),
        });
    }
    // Moving pointers need no snapshot work until a ray actually repeats.
    let stable_ray = rays.iter().any(|ray| {
        cache
            .rays
            .get(&ray.id)
            .is_some_and(|previous| previous.source == *ray)
    });
    let mut reuse = false;
    if stable_ray {
        cache.scratch.clear();
        for (inherited, view, bounds, transform, entity) in &ray_cast.culling_query {
            let visible = match next_settings.1 {
                RayCastVisibility::Any => true,
                RayCastVisibility::Visible => inherited.get(),
                RayCastVisibility::VisibleInView => view.get(),
            };
            let Ok((mesh2d, mesh3d, simplified, backfaces, _)) = ray_cast.mesh_query.get(entity)
            else {
                continue;
            };
            cache.scratch.push(CandidateSnapshot {
                entity,
                visible,
                bounds: *bounds,
                transform: *transform,
                pickable: pickables.get(entity).ok().copied(),
                layers: layers.get(entity).ok().cloned(),
                mesh2d: mesh2d.map(|mesh| mesh.0.id()),
                mesh3d: mesh3d.map(|mesh| mesh.0.id()),
                simplified: simplified.map(|mesh| mesh.0.id()),
                backfaces,
            });
        }
        // Asset resource change detection also catches direct mesh edits before
        // the asset-message schedule has published their Modified receipt.
        reuse = cache.settings == Some(next_settings)
            && cache.candidates == cache.scratch
            && !ray_cast.meshes.is_changed();
        let cache = &mut *cache;
        core::mem::swap(&mut cache.candidates, &mut cache.scratch);
        cache.settings = Some(next_settings);
    } else {
        cache.settings = None;
        cache.candidates.clear();
    }
    let mut previous = core::mem::take(&mut cache.rays);
    for source in rays {
        let ray_id = source.id;
        let ray = source.ray;
        let cam_layers = &source.layers;
        if reuse
            && let Some(cached) = previous.remove(&ray_id)
            && cached.source == source
        {
            if !cached.picks.is_empty() {
                pointer_hits_writer.write(PointerHits::new(
                    ray_id.pointer,
                    cached.picks.clone(),
                    source.order,
                ));
            }
            cache.rays.insert(ray_id, cached);
            continue;
        }
        let settings = MeshRayCastSettings {
            visibility: backend_settings.ray_cast_visibility,
            filter: &|entity| {
                let marker_requirement =
                    !backend_settings.require_markers || marked_targets.get(entity).is_ok();

                // Other entities missing render layers are on the default layer 0
                let entity_layers = layers.get(entity).cloned().unwrap_or_default();
                let render_layers_match = cam_layers.intersects(&entity_layers);

                let is_pickable = pickables.get(entity).ok().is_none_or(|p| p.is_hoverable);

                marker_requirement && render_layers_match && is_pickable
            },
            early_exit_test: &|entity_hit| {
                pickables
                    .get(entity_hit)
                    .is_ok_and(|pickable| pickable.should_block_lower)
            },
        };
        let picks = ray_cast
            .cast_ray(ray, &settings)
            .iter()
            .map(|(entity, hit)| {
                let hit_data = HitData::new(
                    ray_id.camera,
                    hit.distance,
                    Some(hit.point),
                    Some(hit.normal),
                );
                (*entity, hit_data)
            })
            .collect::<Vec<_>>();
        if !picks.is_empty() {
            pointer_hits_writer.write(PointerHits::new(
                ray_id.pointer,
                picks.clone(),
                source.order,
            ));
        }
        cache.rays.insert(ray_id, CachedRay { source, picks });
    }
}

#[cfg(test)]
mod repeat_work_tests {
    use super::*;
    use crate::pointer::PointerId;
    use bevy_asset::{Assets, RenderAssetUsages};
    use bevy_camera::visibility::{InheritedVisibility, ViewVisibility};
    use bevy_ecs::message::{MessageCursor, Messages};
    use bevy_math::{Dir3, Vec3, Vec3A};
    use bevy_mesh::{Indices, Mesh3d, PrimitiveTopology};
    use ray_cast::RayCastBackfaces;

    #[test]
    fn repeat_work_picking_republishes_and_invalidates_exact_hits() {
        let mut app = App::new();
        app.add_plugins(TaskPoolPlugin::default())
            .init_resource::<Assets<Mesh>>()
            .init_resource::<RayMap>()
            .add_message::<PointerHits>()
            .add_plugins(MeshPickingPlugin);
        let camera = app.world_mut().spawn(Camera::default()).id();
        let mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::MAIN_WORLD,
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![[-1., -1., 0.], [1., -1., 0.], [0., 1., 0.]],
        )
        .with_inserted_indices(Indices::U32(vec![0, 1, 2]));
        let mesh = app.world_mut().resource_mut::<Assets<Mesh>>().add(mesh);
        let view = ViewVisibility::VISIBLE;
        let target = app
            .world_mut()
            .spawn((
                Mesh3d(mesh.clone()),
                Aabb {
                    center: Vec3A::ZERO,
                    half_extents: Vec3A::new(1., 1., 0.01),
                },
                GlobalTransform::IDENTITY,
                InheritedVisibility::VISIBLE,
                view,
                RayCastBackfaces,
            ))
            .id();
        app.world_mut().resource_mut::<RayMap>().map.insert(
            RayId::new(camera, PointerId::Mouse),
            Ray3d::new(Vec3::new(0., 0., 2.), Dir3::NEG_Z),
        );
        let mut cursor = MessageCursor::<PointerHits>::default();
        let mut step = |app: &mut App| {
            app.update();
            cursor
                .read(app.world().resource::<Messages<PointerHits>>())
                .flat_map(|hits| hits.picks.iter().map(|(entity, _)| *entity))
                .collect::<Vec<_>>()
        };
        for _ in 0..4 {
            assert_eq!(step(&mut app), vec![target]);
        }
        app.world_mut()
            .entity_mut(target)
            .insert(GlobalTransform::from_translation(Vec3::X * 10.));
        assert!(step(&mut app).is_empty());
        app.world_mut()
            .entity_mut(target)
            .insert((GlobalTransform::IDENTITY, Pickable::IGNORE));
        assert!(step(&mut app).is_empty());
        app.world_mut().entity_mut(target).remove::<Pickable>();
        assert_eq!(step(&mut app), vec![target]);
        for _ in 0..3 {
            assert_eq!(step(&mut app), vec![target]);
        }
        app.world_mut()
            .resource_mut::<Assets<Mesh>>()
            .get_mut(&mesh)
            .unwrap()
            .insert_attribute(
                Mesh::ATTRIBUTE_POSITION,
                vec![[9., -1., 0.], [11., -1., 0.], [10., 1., 0.]],
            );
        assert!(step(&mut app).is_empty());
        app.world_mut().despawn(target);
        assert!(step(&mut app).is_empty());
    }
}

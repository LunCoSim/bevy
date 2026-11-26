use bevy_material::bind_group_layout_entries::{
    binding_types::{storage_buffer_read_only, uniform_buffer_sized},
    BindGroupLayoutEntryBuilder,
};

use super::BufferVec;
use crate::{
    render_resource::batched_uniform_buffer::BatchedUniformBuffer,
    renderer::{RenderDevice, RenderQueue},
};
use bevy_ecs::{prelude::Component, resource::Resource};
use core::{
    borrow::Borrow,
    marker::PhantomData,
    ops::Range,
    sync::atomic::{AtomicU64, Ordering},
};
use encase::{private::WriteInto, ShaderSize, ShaderType};
use nonmax::NonMaxU32;
use wgpu::{BindingResource, BufferUsages, Limits};

/// Trait for types able to go in a [`GpuArrayBuffer`].
pub trait GpuArrayBufferable: ShaderType + ShaderSize + WriteInto + Clone {}

impl<T: ShaderType + ShaderSize + WriteInto + Clone> GpuArrayBufferable for T {}

/// A per-instance value that can be encoded by reference, or moved into the
/// temporary array when storage buffers are unavailable.
/// Providers that already retain frame data return a reference; providers
/// computing a fresh value return the value itself.
pub trait GpuArrayElement<T: GpuArrayBufferable>: Borrow<T> {
    /// Obtain owned data for the dynamic-uniform array path.
    fn into_owned(self) -> T;

    /// Identity of immutable shader data, when retained by the provider.
    /// Equal revisions must represent identical shader-layout bytes.
    fn revision(&self) -> Option<GpuArrayElementRevision> {
        None
    }
}

/// A process-wide identity for one immutable prepared element version.
/// Allocate a new revision whenever any of its encoded fields changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuArrayElementRevision(u64);

impl GpuArrayElementRevision {
    pub fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(
            NEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                .expect("GPU array element revision space exhausted"),
        )
    }
}

/// Borrows a prepared element together with the version of its shader data.
pub struct RetainedGpuArrayElement<'a, T> {
    value: &'a T,
    revision: GpuArrayElementRevision,
}

impl<'a, T> RetainedGpuArrayElement<'a, T> {
    pub fn new(value: &'a T, revision: GpuArrayElementRevision) -> Self {
        Self { value, revision }
    }
}

impl<T> Borrow<T> for RetainedGpuArrayElement<'_, T> {
    fn borrow(&self) -> &T {
        self.value
    }
}

impl<T: GpuArrayBufferable> GpuArrayElement<T> for RetainedGpuArrayElement<'_, T> {
    fn into_owned(self) -> T {
        self.value.clone()
    }

    fn revision(&self) -> Option<GpuArrayElementRevision> {
        Some(self.revision)
    }
}

/// Retains the shader-layout bytes and their versions at the existing batch
/// indices. Each frame still appends in native phase order. A slot whose exact
/// version is unchanged requires no encoding. Uploads cover the bounding extent
/// of dirty slots, avoiding independent queue writes for disjoint edits.
pub struct RetainedStorageBuffer<T: GpuArrayBufferable> {
    pub(super) buffer: BufferVec<T>,
    revisions: Vec<Option<GpuArrayElementRevision>>,
    cursor: usize,
    dirty: Option<Range<usize>>,
}

impl<T: GpuArrayBufferable> RetainedStorageBuffer<T> {
    pub fn new() -> Self {
        Self {
            buffer: BufferVec::new(BufferUsages::STORAGE),
            revisions: Vec::new(),
            cursor: 0,
            dirty: None,
        }
    }

    fn clear(&mut self) {
        self.cursor = 0;
    }

    fn push(&mut self, value: impl GpuArrayElement<T>) -> usize {
        let index = self.cursor;
        self.cursor += 1;
        let revision = value.revision();
        if index == self.revisions.len() {
            self.buffer.push_ref(value.borrow());
            self.revisions.push(revision);
        } else if revision.is_some() && self.revisions[index] == revision {
            return index;
        } else {
            self.buffer.set_ref(index, value.borrow());
            self.revisions[index] = revision;
        }
        match &mut self.dirty {
            Some(range) => {
                range.start = range.start.min(index);
                range.end = range.end.max(index + 1);
            }
            None => self.dirty = Some(index..index + 1),
        }
        index
    }

    fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        self.revisions.truncate(self.cursor);
        self.buffer.truncate(self.cursor);
        let dirty = self.dirty.take();
        if self.cursor == 0 {
            return;
        }
        if self.cursor > self.buffer.capacity() || self.buffer.buffer().is_none() {
            // A new allocation has no resident bytes, even for unchanged slots.
            self.buffer.write_buffer(device, queue);
        } else if let Some(range) = dirty {
            let end = range.end.min(self.cursor);
            if range.start < end {
                self.buffer
                    .write_buffer_range(queue, range.start..end)
                    .expect("retained instance range must be allocated");
            }
        }
    }

    fn binding(&self) -> Option<BindingResource<'_>> {
        self.buffer.binding()
    }
}

impl<T: GpuArrayBufferable> Default for RetainedStorageBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: GpuArrayBufferable> GpuArrayElement<T> for T {
    #[inline]
    fn into_owned(self) -> T {
        self
    }
}

impl<T: GpuArrayBufferable> GpuArrayElement<T> for &T {
    #[inline]
    fn into_owned(self) -> T {
        self.clone()
    }
}

/// Stores an array of elements to be transferred to the GPU and made accessible to shaders as a read-only array.
///
/// On platforms that support storage buffers, this retains shader-layout
/// bytes at their existing batch indices. Otherwise, this falls back to a dynamic offset
/// uniform buffer with the largest array of T that fits within a uniform buffer
/// binding (within reasonable limits).
///
/// Other options for storing GPU-accessible data are:
/// * [`BufferVec`]
/// * [`DynamicStorageBuffer`](crate::render_resource::DynamicStorageBuffer)
/// * [`DynamicUniformBuffer`](crate::render_resource::DynamicUniformBuffer)
/// * [`RawBufferVec`](crate::render_resource::RawBufferVec)
/// * [`StorageBuffer`](crate::render_resource::StorageBuffer)
/// * [`Texture`](crate::render_resource::Texture)
/// * [`UniformBuffer`](crate::render_resource::UniformBuffer)
#[derive(Resource)]
pub enum GpuArrayBuffer<T: GpuArrayBufferable> {
    Uniform(BatchedUniformBuffer<T>),
    Storage(RetainedStorageBuffer<T>),
}

impl<T: GpuArrayBufferable> GpuArrayBuffer<T> {
    pub fn new(limits: &Limits) -> Self {
        if limits.max_storage_buffers_per_shader_stage == 0 {
            GpuArrayBuffer::Uniform(BatchedUniformBuffer::new(limits))
        } else {
            GpuArrayBuffer::Storage(RetainedStorageBuffer::new())
        }
    }

    pub fn clear(&mut self) {
        match self {
            GpuArrayBuffer::Uniform(buffer) => buffer.clear(),
            GpuArrayBuffer::Storage(buffer) => buffer.clear(),
        }
    }

    pub fn push(&mut self, value: impl GpuArrayElement<T>) -> GpuArrayBufferIndex<T> {
        match self {
            GpuArrayBuffer::Uniform(buffer) => buffer.push(value.into_owned()),
            GpuArrayBuffer::Storage(buffer) => {
                let index = buffer.push(value) as u32;
                GpuArrayBufferIndex {
                    index,
                    dynamic_offset: None,
                    element_type: PhantomData,
                }
            }
        }
    }

    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        match self {
            GpuArrayBuffer::Uniform(buffer) => buffer.write_buffer(device, queue),
            GpuArrayBuffer::Storage(buffer) => buffer.write_buffer(device, queue),
        }
    }

    pub fn binding_layout(limits: &Limits) -> BindGroupLayoutEntryBuilder {
        if limits.max_storage_buffers_per_shader_stage == 0 {
            uniform_buffer_sized(
                true,
                // BatchedUniformBuffer uses a MaxCapacityArray that is runtime-sized, so we use
                // None here and let wgpu figure out the size.
                None,
            )
        } else {
            storage_buffer_read_only::<T>(false)
        }
    }

    pub fn binding(&self) -> Option<BindingResource<'_>> {
        match self {
            GpuArrayBuffer::Uniform(buffer) => buffer.binding(),
            GpuArrayBuffer::Storage(buffer) => buffer.binding(),
        }
    }

    pub fn batch_size(limits: &Limits) -> Option<u32> {
        if limits.max_storage_buffers_per_shader_stage == 0 {
            Some(BatchedUniformBuffer::<T>::batch_size(limits) as u32)
        } else {
            None
        }
    }
}

/// An index into a [`GpuArrayBuffer`] for a given element.
#[derive(Component, Clone)]
pub struct GpuArrayBufferIndex<T: GpuArrayBufferable> {
    /// The index to use in a shader into the array.
    pub index: u32,
    /// The dynamic offset to use when setting the bind group in a pass.
    /// Only used on platforms that don't support storage buffers.
    pub dynamic_offset: Option<NonMaxU32>,
    pub element_type: PhantomData<T>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_retained_storage_tracks_exact_slot_revision_and_dirty_extent() {
        let mut storage = RetainedStorageBuffer::<u32>::new();
        let a = GpuArrayElementRevision::next();
        let b = GpuArrayElementRevision::next();
        let c = GpuArrayElementRevision::next();
        assert_eq!(storage.push(RetainedGpuArrayElement::new(&11, a)), 0);
        assert_eq!(storage.push(RetainedGpuArrayElement::new(&22, b)), 1);
        assert_eq!(storage.push(RetainedGpuArrayElement::new(&33, c)), 2);
        assert_eq!(storage.dirty.take(), Some(0..3));

        storage.clear();
        storage.push(RetainedGpuArrayElement::new(&11, a));
        storage.push(RetainedGpuArrayElement::new(&22, b));
        storage.push(RetainedGpuArrayElement::new(&33, c));
        assert_eq!(storage.dirty, None);

        storage.clear();
        storage.push(RetainedGpuArrayElement::new(&11, a));
        let changed = GpuArrayElementRevision::next();
        storage.push(RetainedGpuArrayElement::new(&44, changed));
        storage.push(RetainedGpuArrayElement::new(&33, c));
        assert_eq!(storage.dirty.take(), Some(1..2));

        storage.clear();
        storage.push(RetainedGpuArrayElement::new(&33, c));
        storage.push(RetainedGpuArrayElement::new(&44, changed));
        storage.push(RetainedGpuArrayElement::new(&11, a));
        assert_eq!(storage.dirty.take(), Some(0..3));

        // Unversioned providers still encode their fresh values every frame.
        storage.clear();
        storage.push(33);
        assert_eq!(storage.dirty.take(), Some(0..1));
        storage.clear();
        storage.push(33);
        assert_eq!(storage.dirty.take(), Some(0..1));
    }
}

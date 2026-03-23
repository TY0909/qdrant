use std::borrow::Cow;
use std::path::{Path, PathBuf};

use super::file_ops::UniversalReadFileOps;
use super::mmap::MmapFile;
use super::{OpenOptions, ReadRange, Result, UniversalRead};
use crate::generic_consts::AccessPattern;

/// Runtime dispatch between mmap and disk-cached storage.
///
/// When the global [`CacheController`](crate::universal_io::disk_cache::CacheController)
/// is initialized, files are opened through the block cache
/// ([`CachedSlice`](crate::universal_io::disk_cache::CachedSlice)). Otherwise falls back
/// to [`MmapFile`].
pub enum StorageEnum<T: Copy + 'static> {
    Mmap(MmapFile),
    #[cfg(target_os = "linux")]
    Cached(crate::universal_io::disk_cache::CachedSlice<T>),
    #[cfg(not(target_os = "linux"))]
    _Phantom(std::marker::PhantomData<T>),
}

impl<T: Copy + 'static> std::fmt::Debug for StorageEnum<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mmap(_) => f.debug_tuple("StorageEnum::Mmap").finish(),
            #[cfg(target_os = "linux")]
            Self::Cached(_) => f.debug_tuple("StorageEnum::Cached").finish(),
            #[cfg(not(target_os = "linux"))]
            Self::_Phantom(_) => unreachable!(),
        }
    }
}

impl<T: Copy + 'static> UniversalReadFileOps for StorageEnum<T> {
    fn list_files(prefix_path: &Path) -> Result<Vec<PathBuf>> {
        MmapFile::list_files(prefix_path)
    }

    fn exists(path: &Path) -> Result<bool> {
        <MmapFile as UniversalReadFileOps>::exists(path)
    }
}

impl<T: bytemuck::Pod> UniversalRead<T> for StorageEnum<T> {
    fn open(path: impl AsRef<Path>, options: OpenOptions) -> Result<Self>
    where
        Self: Sized,
    {
        #[cfg(target_os = "linux")]
        if crate::universal_io::disk_cache::CacheController::global().is_some() {
            let cached = <crate::universal_io::disk_cache::CachedSlice<T> as UniversalRead<T>>::open(
                path, options,
            )?;
            return Ok(Self::Cached(cached));
        }

        let mmap = <MmapFile as UniversalRead<T>>::open(path, options)?;
        Ok(Self::Mmap(mmap))
    }

    fn read<P: AccessPattern>(&self, range: ReadRange) -> Result<Cow<'_, [T]>> {
        match self {
            Self::Mmap(m) => <MmapFile as UniversalRead<T>>::read::<P>(m, range),
            #[cfg(target_os = "linux")]
            Self::Cached(c) => c.read::<P>(range),
            #[cfg(not(target_os = "linux"))]
            Self::_Phantom(_) => unreachable!(),
        }
    }

    fn read_whole(&self) -> Result<Cow<'_, [T]>> {
        match self {
            Self::Mmap(m) => <MmapFile as UniversalRead<T>>::read_whole(m),
            #[cfg(target_os = "linux")]
            Self::Cached(c) => <_ as UniversalRead<T>>::read_whole(c),
            #[cfg(not(target_os = "linux"))]
            Self::_Phantom(_) => unreachable!(),
        }
    }

    fn read_batch<'a, P: AccessPattern, Meta: 'a>(
        &'a self,
        ranges: impl IntoIterator<Item = (Meta, ReadRange)>,
        callback: impl FnMut(Meta, &[T]) -> Result<()>,
    ) -> Result<()> {
        match self {
            Self::Mmap(m) => {
                <MmapFile as UniversalRead<T>>::read_batch::<P, Meta>(m, ranges, callback)
            }
            #[cfg(target_os = "linux")]
            Self::Cached(c) => c.read_batch::<P, Meta>(ranges, callback),
            #[cfg(not(target_os = "linux"))]
            Self::_Phantom(_) => unreachable!(),
        }
    }

    fn len(&self) -> Result<u64> {
        match self {
            Self::Mmap(m) => <MmapFile as UniversalRead<T>>::len(m),
            #[cfg(target_os = "linux")]
            Self::Cached(c) => <_ as UniversalRead<T>>::len(c),
            #[cfg(not(target_os = "linux"))]
            Self::_Phantom(_) => unreachable!(),
        }
    }

    fn populate(&self) -> Result<()> {
        match self {
            Self::Mmap(m) => <MmapFile as UniversalRead<T>>::populate(m),
            #[cfg(target_os = "linux")]
            Self::Cached(c) => <_ as UniversalRead<T>>::populate(c),
            #[cfg(not(target_os = "linux"))]
            Self::_Phantom(_) => unreachable!(),
        }
    }

    fn clear_ram_cache(&self) -> Result<()> {
        match self {
            Self::Mmap(m) => <MmapFile as UniversalRead<T>>::clear_ram_cache(m),
            #[cfg(target_os = "linux")]
            Self::Cached(c) => <_ as UniversalRead<T>>::clear_ram_cache(c),
            #[cfg(not(target_os = "linux"))]
            Self::_Phantom(_) => unreachable!(),
        }
    }
}

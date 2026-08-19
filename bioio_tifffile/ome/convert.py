#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Direct QPTIFF -> OME-Zarr conversion.

The primary entry point for batch pipelines: no bioio ``Reader`` in the path,
just parse -> arrays -> assemble -> serialise.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, MutableMapping, Optional, Union

log = logging.getLogger(__name__)


def qptiff_to_ome_zarr(
    src: Union[str, Path],
    dst: Union[str, Path, MutableMapping[str, Any]],
    *,
    overwrite: bool = False,
    ome_only: bool = False,
    scene: Optional[str] = None,
    **write_kwargs: Any,
) -> Any:
    """
    Convert a QPTIFF straight to a bioformats2raw-layout OME-Zarr store.

    Parameters
    ----------
    src:
        Path to the .qptiff.
    dst:
        Destination store (path or zarr-compatible mapping).
    overwrite:
        Replace an existing store.
    ome_only:
        Omit the vendor ``qpi`` attribute block, keeping only spec metadata.
    scene:
        Scene to convert; defaults to the reader's first scene.
    **write_kwargs:
        Forwarded to :func:`~bioio_tifffile.ome.zarr_writer.write_ome_zarr`
        (``chunk_shape``, ``shard_shape``, ``compressor``, ``validate``, ...).

    Returns
    -------
    zarr.Group
        The written fileset root.

    Examples
    --------
    Worker-script shape, with dask sized from the SLURM allocation::

        import os, dask
        from bioio_tifffile.ome import qptiff_to_ome_zarr

        dask.config.set(
            scheduler="threads",
            num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)),
        )
        qptiff_to_ome_zarr(src, dst, overwrite=True)
    """
    # imported here so the module stays cheap and the dependency direction
    # stays one-way: reader -> ome, never ome -> reader at import time.
    from ..reader import Reader
    from .model import ome_metadata_from_qptiff
    from .zarr_writer import write_ome_zarr

    reader = Reader(str(src))
    if scene is not None:
        reader.set_scene(scene)

    datatree = reader.xarray_dask_datatree_data
    meta = reader.qpi_metadata

    level0 = datatree["scale0"].ds["image"]
    dims = {d: s for d, s in zip(level0.dims, level0.shape)}

    ome = ome_metadata_from_qptiff(
        meta,
        scene_name=reader.current_scene,
        size_x=dims.get("x"),
        size_y=dims.get("y"),
        size_c=dims.get("c"),
        dtype_name=str(level0.dtype),
    )

    log.info("converting %s -> %s (%d channels)", src, dst, len(meta.channels))
    return write_ome_zarr(
        datatree,
        ome,
        dst,
        overwrite=overwrite,
        ome_only=ome_only,
        meta=meta,
        **write_kwargs,
    )

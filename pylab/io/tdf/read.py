"""tdf file reading module"""


__all__ = ["read_tdf"]


#! IMPORTS


import os
import struct
from io import BufferedReader
from typing import Any, Literal

import numpy as np
import pandas as pd


#! CONSTANTS


_BLOCK_KEYS = ["Type", "Format", "Offset", "Size"]


#! FUNCTIONS


def _open_tdf(
    path: str,
):
    """
    open a tdf file and return the BufferedReader object, the blocks contained,
    and the file version.

    Parameters
    ----------
    path: str
        an existing tdf path.

    Returns
    -------
    fid: BufferedReader
        the file stream object

    blocks: list[dict[str, int]]
        the list of blocks info

    version: int
        the file version.
    """
    tdf_signature = "41604B82CA8411D3ACB60060080C6816"
    blocks: list[dict[Literal["Type", "Format", "Offset", "Size"], int]] = []
    next_entry_offset = 40
    version = float("nan")

    # check the validity of the entered path
    assert os.path.exists(path), path + " does not exist."
    assert path[-4:] == ".tdf", path + ' must be an ".tdf" path.'

    # try opening the file
    fid = open(path, "rb")
    try:
        # check the signature
        sig = struct.unpack("IIII", fid.read(16))
        sig = "".join([f"{b:08x}" for b in sig])
        if sig.upper() != tdf_signature:
            raise IOError("invalid file")

        # get the number of entries
        version, n_entries = struct.unpack("Ii", fid.read(8))
        assert n_entries > 0, "The file specified contains no data."

        # check each entry to find the available blocks
        for _ in range(n_entries):
            if -1 == fid.seek(next_entry_offset, 1):
                raise IOError("Error: the file specified is corrupted.")

            # get the data types
            block_info = struct.unpack("IIii", fid.read(16))
            if block_info[1] != 0:  # Format != 0 ensures valid blocks
                blocks += [dict(zip(_BLOCK_KEYS, block_info))]  # type: ignore

            # update the offset
            next_entry_offset = 272

    except Exception as exc:
        raise RuntimeError(exc) from exc

    return fid, blocks, int(version)


def _get_block(
    fid: BufferedReader,
    blocks: list[dict[Literal["Type", "Format", "Offset", "Size"], int]],
    block_id: int,
):
    """
    return the blocks according to the provided id

    Parameters
    ----------
    fid: BufferedReader
        the file stream object

    blocks : list[dict[str, int]]
        the blocks_info extracted by the _open_tdf function

    block_id : int
        the required block id

    Returns
    -------
    fid: BufferedReader
        the file stream object

    valid: dict[Literal["Type", "Format", "Offset", "Size"], int],
        the list of valid blocks
    """
    block = [i for i in blocks if block_id == i["Type"]]
    if len(block) == 0 or -1 == fid.seek(block[0]["Offset"], 0):
        fid.close()
    return fid, block[0]


def _calibration_generic(
    fid: BufferedReader,
    blocks: list[dict[Literal["Type", "Format", "Offset", "Size"], int]],
):
    """
    read calibration data for general purpose

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks : list[dict[Literal["Type", "Format", "Offset", "Size"], int]]
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    calibration_data: dict[str, np.ndarray[Any, np.dtype[np.float_]]]
        the available calibration data.
    """

    # get the block and the number of signals
    fid, block = _get_block(fid, blocks, 15)
    if len(block) == 0:
        return None
    sig_n = struct.unpack("i", fid.read(4))[0]
    fid.seek(4, 1)

    # ge tthe channels map
    sig_map = np.array(list(struct.unpack(f"{sig_n}h", fid.read(2 * sig_n))))

    # get the calibration data
    sig_cal = np.nan * np.ones((sig_n, 3))
    for i in np.arange(sig_n):
        sig_cal[i, 0] = struct.unpack("i", fid.read(4))[0]
        sig_cal[i, 1:] = struct.unpack("ff", fid.read(8))

    return {
        "CHANNELS": sig_map,
        "DEVICE_TYPE": sig_cal[:, 0],
        "M": sig_cal[:, 1],
        "Q": sig_cal[:, 2],
    }


def _camera_calibration(
    fid: BufferedReader,
    blocks: list[dict[Literal["Type", "Format", "Offset", "Size"], int]],
):
    """
    read calibration data for general purpose

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks : list[dict[Literal["Type", "Format", "Offset", "Size"], int]]
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    calibration_data: dict[str, Any]
        the available calibration data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 2)
    if len(block) == 0:
        return None
    cam_n = np.array(struct.unpack("1i", fid.read(4)))[0]  # number of cams
    cam_m = np.array(struct.unpack("1I", fid.read(4)))[0]  # model
    cam_d = np.array(struct.unpack("3f", fid.read(12)))  # dimensions
    cam_r = np.reshape(struct.unpack("9f", fid.read(36)), (3, 3))  # rot mat
    cam_t = np.array(struct.unpack("3f", fid.read(12)))  # translation
    if cam_m == 0:
        cam_m = "none"
    elif cam_m == 1:
        cam_m = "kali"
    elif cam_m == 2:
        cam_m = "amass"
    elif cam_m == 3:
        cam_m = "thor"
    else:
        raise ValueError("cam_m value not recognized")

    # channels map
    cam_map = np.array(struct.unpack(f"{cam_n}h", fid.read(2 * cam_n)))

    # parameters
    cam_params = []
    for i in np.arange(cam_n):
        if 1 == block["Format"]:  # Seelab type 1 calibration
            params = {
                "R": np.reshape(struct.unpack("9d", fid.read(72)), (3, 3)),
                "T": np.array(struct.unpack("3d", fid.read(24))),
                "F": np.array(struct.unpack("2d", fid.read(16))),
                "C": np.array(struct.unpack("2d", fid.read(16))),
                "K1": np.array(struct.unpack("2d", fid.read(16))),
                "K2": np.array(struct.unpack("2d", fid.read(16))),
                "K3": np.array(struct.unpack("2d", fid.read(16))),
                "VP": np.reshape(struct.unpack("4i", fid.read(16)), (2, 2)),
                # origin = VP[:, 0] size = VP[:, 1]
            }
        elif 2 == block["Format"]:  # BTS
            params = {
                "R": np.reshape(struct.unpack("9d", fid.read(72)), (3, 3)),
                "T": np.array(struct.unpack("3d", fid.read(24))),
                "F": np.array(struct.unpack("1d", fid.read(16))),
                "C": np.array(struct.unpack("2d", fid.read(16))),
                "KX": np.array(struct.unpack("70d", fid.read(560))),
                "KY": np.array(struct.unpack("70d", fid.read(560))),
                "VP": np.reshape(struct.unpack("4i", fid.read(16)), (2, 2)),
                # origin = VP[:, 0] size = VP[:, 1]
            }
        else:
            msg = f"block['Format'] must be 1 or 2, but {block['Format']}"
            msg += " was found."
            raise ValueError(msg)
        cam_params += [params]

    return {
        "DIMENSION": cam_d,
        "ROTATION_MATRIX": cam_r,
        "TRANSLATION": cam_t,
        "MODEL": cam_m,
        "CHANNELS": cam_map,
        "PARAMS": cam_params,
    }


def _read_frames_2d_rts(
    fid: BufferedReader,
    nframes: int,
    ncams: int,
):
    """
    read frames from 2D data according to the RTS (real time stream) format.

    Parameters
    ----------
    fid : BufferedReader
        file stream

    nframes : int
        number of available frames

    ncams : int
        number of available cams

    Returns
    -------
    frames: list
        a list of frames having shape (nframes, ncams, 2, nfeats)
    """
    frames = []
    for _ in np.arange(nframes):
        frame = []
        for _ in np.arange(ncams):
            nfeats = np.array(struct.unpack("i", fid.read(4)))[0]
            fid.seek(4, 1)
            vals = struct.unpack("f" * 2 * nfeats, fid.read(8 * nfeats))
            frame += [np.reshape(vals, (2, nfeats)).tolist()]
        frames += [frame]
    return frames


def _read_frames_2d_pck(
    fid: BufferedReader,
    nframes: int,
    ncams: int,
):
    """
    read frames from 2D data according to the PCK (packed data) format.

    Parameters
    ----------
    fid : BufferedReader
        file stream

    nframes : int
        number of available frames

    ncams : int
        number of available cams

    Returns
    -------
    frames: list
        a list of frames having shape (nframes, ncams, 2, nfeats)
    """
    nsamp = int(ncams * nframes)
    nfeats = struct.unpack(f"{nsamp}h", fid.read(2 * nsamp))
    nfeats = np.reshape(nfeats, (ncams, nframes))
    frames = []
    for frm in np.arange(nframes):
        frame = []
        for cam in np.arange(ncams):
            num = int(2 * nfeats[cam, frm])
            vals = struct.unpack(f"{num}f", fid.read(4 * num))
            frame += [np.reshape(vals, (2, nfeats[cam, frm])).tolist()]
        frames += [frame]
    return frames


def _read_frames_2d_syn(
    fid: BufferedReader,
    nframes: int,
    ncams: int,
):
    """
    read frames from 2D data according to the SYN (sync data) format.

    Parameters
    ----------
    fid : BufferedReader
        file stream

    nframes : int
        number of available frames

    ncams : int
        number of available cams

    Returns
    -------
    frames: list
        a list of frames having shape (nframes, ncams, 2, nfeats)
    """
    max_feats_n = np.array(struct.unpack("1h", fid.read(2)))[0]
    shape = (nframes, ncams)
    nsamp = int(np.prod(shape))
    nfeats = struct.unpack(f"{nsamp}h", fid.read(2 * nsamp))
    nfeats = np.reshape(nfeats, shape)
    frames = []
    for f in np.arange(nframes):
        frame = []
        for c in np.arange(ncams):
            nsamp = 2 * max_feats_n
            tmp_buf = struct.unpack(f"{nsamp}f", fid.read(4 * nsamp))
            tmp_buf = np.reshape(tmp_buf, (2, max_feats_n))
            frame += [tmp_buf[:, : nfeats[c, f]].tolist()]
        frames += [frame]
    return frames


def _data_2d(
    fid: BufferedReader,
    blocks: list[dict[Literal["Type", "Format", "Offset", "Size"], int]],
):
    """
    read 2D data sequence from a tdf file stream.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks : list[dict[Literal["Type", "Format", "Offset", "Size"], int]]
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 4)
    if len(block) == 0:
        return None
    ncams, nframes, freq, time0 = struct.unpack("iiif", fid.read(16))
    fid.seek(4, 1)

    # channels map
    cam_map = np.array(struct.unpack(f"{ncams}h", fid.read(2 * ncams)))

    # features extraction
    if 1 == block["Format"]:  # RTS: Real Time Stream
        read_frames = _read_frames_2d_rts
    elif 2 == block["Format"]:  # PCK: Packed Data format
        read_frames = _read_frames_2d_pck
    elif 3 == block["Format"]:  # SYNC: Synchronized Data format
        read_frames = _read_frames_2d_syn
    else:
        msg = f"block['Format'] must be 1, 2 or 3, but {block['Format']}"
        msg += " was found."
        raise ValueError(msg)

    return {
        "CAMS": ncams,
        "FRAMES": nframes,
        "FREQUENCY": freq,
        "TIME0": time0,
        "CHANNELS": cam_map,
        "FEATURES": read_frames(fid, nframes, ncams),
    }


def _data_2d_camera_calibration(
    fid: BufferedReader,
    blocks: list[dict[Literal["Type", "Format", "Offset", "Size"], int]],
):
    """
    read 2D data sequence from a tdf file stream to be used
    for camera calibration.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks : list[dict[Literal["Type", "Format", "Offset", "Size"], int]]
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 3)
    if len(block) == 0:
        return None
    ncams, naxesframes, nwandframes, freq = struct.unpack("iiii", fid.read(16))
    fid.seek(4, 1)
    axespars = np.array(struct.unpack("9f", fid.read(36)))
    wandpars = np.array(struct.unpack("2f", fid.read(8)))

    # channels map
    cam_map = np.array(struct.unpack(f"{ncams}h", fid.read(2 * ncams)))

    # features extraction function
    if 1 == block["Format"]:  # RTS: Real Time Stream
        read_frames = _read_frames_2d_rts
    elif 2 == block["Format"]:  # PCK: Packed Data format
        read_frames = _read_frames_2d_pck
    elif 3 == block["Format"]:  # SYNC: Synchronized Data format
        read_frames = _read_frames_2d_syn
    else:
        msg = f"block['Format'] must be 1, 2 or 3, but {block['Format']}"
        msg += " was found."
        raise ValueError(msg)

    return {
        "CAMS": ncams,
        "FREQUENCY": freq,
        "CHANNELS": cam_map,
        "AXES_FEATURES": read_frames(fid, naxesframes, ncams),
        "WAND_FEATURES": read_frames(fid, nwandframes, ncams),
        "AXES_PARAMS": axespars,
        "WAND_PARAMS": wandpars,
    }


def _data_2d_platforms_calibration(
    fid: BufferedReader,
    blocks: list[dict[Literal["Type", "Format", "Offset", "Size"], int]],
):
    """
    read 2D data sequence from a tdf file stream to be used
    for force platforms calibration.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks : list[dict[Literal["Type", "Format", "Offset", "Size"], int]]
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 8)
    if len(block) == 0:
        return None
    nplats, ncams, freq = struct.unpack("iii", fid.read(16))
    fid.seek(4, 1)

    # channels map
    cam_map = np.array(struct.unpack(f"{ncams}h", fid.read(2 * ncams)))
    plat_map = np.array(struct.unpack(f"{nplats}h", fid.read(2 * nplats)))

    # features extraction function
    if 1 == block["Format"]:  # RTS: Real Time Stream
        read_frames = _read_frames_2d_rts
    elif 2 == block["Format"]:  # PCK: Packed Data format
        read_frames = _read_frames_2d_pck
    elif 3 == block["Format"]:  # SYNC: Synchronized Data format
        read_frames = _read_frames_2d_syn
    else:
        msg = f"block['Format'] must be 1, 2 or 3, but {block['Format']}"
        msg += " was found."
        raise ValueError(msg)

    # read data for each platform
    platforms = []
    keys = ["LABEL", "FRAMES", "SIZE", "FEATURES"]
    for _ in np.arange(nplats):
        lbls = " ".join("".join(struct.unpack("32B", fid.read(32))).split("0"))
        frames = np.array(struct.unpack("i", fid.read(4)))[0]
        size = list(struct.unpack("ff", fid.read(8)))[0]
        feats = read_frames(fid, frames, ncams)
        vals = (lbls, frames, size, feats)
        platforms += [dict(zip(keys, vals))]

    return {
        "CAMS": ncams,
        "PLATFORMS": nplats,
        "FREQUENCY": freq,
        "CAMERA_CHANNELS": cam_map,
        "PLATFORM_CHANNELS": plat_map,
        "PLATFORM_FEATURES": platforms,
    }


def _read_tracks(
    fid: BufferedReader,
    nframes: int,
    ntracks: int,
    nchannels: int,
    haslabels: bool,
):
    """
    read data by track

    Parameters
    ----------
    fid : BufferedReader
        file stream

    nframes : int
        available frames

    ntracks : int
        number of tracks

    nchannels: int
        the number of channels to extract

    haslabels: bool
        if True the track labels are returned

    Returns
    -------
    data: dict[str, np.ndarray]
        the parsed tracks.
    """
    obj = np.ones((ntracks, nframes, nchannels)) * np.nan
    lbls = []
    for trk in np.arange(ntracks):
        # get the labels
        if haslabels:
            lbls += ["".join(struct.unpack("256B", fid.read(256)))]
        else:
            lbls += [f"track{trk + 1}"]

        # get the available segments
        nseg = np.array(struct.unpack("i", fid.read(4)))[0]
        fid.seek(4, 1)
        nsamp = 2 * nseg
        segments = struct.unpack(f"{nsamp}i", fid.read(4 * nsamp))
        segments = np.reshape(segments, (2, nseg))

        # read the data for the actual track
        for sgm in np.arange(nseg):
            for frm in np.arange(segments[0, sgm], segments[1, sgm] + 1):
                vals = fid.read(4 * nchannels)
                obj[trk, frm] = np.array(struct.unpack("f" * nchannels, vals))

    # split data by track
    out: dict[str, np.ndarray[Any, np.dtype[np.float_]]] = dict(zip(lbls, obj))
    return out


def _read_frames(
    fid: BufferedReader,
    nframes: int,
    ntracks: int,
    nchannels: int,
    haslabels: bool = True,
):
    """
    read 3D data by frame

    Parameters
    ----------
    fid : BufferedReader
        file stream

    nframes : int
        available frames

    ntracks : int
        number of tracks

    nchannels: int
        the number of channels to extract

    haslabels: bool
        if True the track labels are returned

    Returns
    -------
    data3d: dict[str, pd.DataFrame]
        the parsed tracks.
    """

    # get the labels
    lbls = []
    for trk in np.arange(ntracks):
        if haslabels:
            label = "".join(struct.unpack("256B", fid.read(256))).strip()
        else:
            label = f"track{trk + 1}"
        lbls += [label]

    # get the available data
    nsamp = nchannels * ntracks * nframes
    data = struct.unpack(f"{nsamp}f", fid.read(4 * nsamp))
    data = np.reshape(data, (ntracks, nframes, nchannels))

    # return
    out: dict[str, np.ndarray[Any, np.dtype[np.float_]]] = dict(zip(lbls, data))
    return out


def _data_3d(
    fid: BufferedReader,
    blocks: list[dict[Literal["Type", "Format", "Offset", "Size"], int]],
):
    """
    read 2D data sequence from a tdf file stream.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks : list[dict[Literal["Type", "Format", "Offset", "Size"], int]]
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 5)
    if len(block) == 0:
        return None
    nframes, freq, time0, ntracks = struct.unpack("iifi", fid.read(16))
    dims = np.array(struct.unpack("3f", fid.read(12)))
    rmat = np.reshape(struct.unpack("9f", fid.read(36)), (3, 3))
    tras = np.array(struct.unpack("3f", fid.read(12)))
    fid.seek(4, 1)

    # get links
    if block["Format"] in [1, 3]:
        nlinks = np.array(struct.unpack("i", fid.read(4)))[0]
        fid.seek(4, 1)
        nsamp = 2 * nlinks
        links = struct.unpack(f"{nsamp}i", fid.read(nsamp * 4))
        links = np.reshape(links, (2, nlinks))
    else:
        links = np.array([])

    # get the data
    if block["Format"] in [1, 2]:
        tracks = _read_tracks(fid, nframes, ntracks, 3, True)
    elif block["Format"] in [3, 4]:
        tracks = _read_frames(fid, nframes, ntracks, 3, True)
    else:
        msg = f"block['Format'] must be 1, 2, 3 or 4, but {block['Format']}"
        msg += " was found."
        raise ValueError(msg)

    return {
        "TRACKS": tracks,
        "LINKS": links,
        "FREQUENCY": freq,
        "TIME0": time0,
        "DIMENSIONS": dims,
        "ROTATION_MATRIX": rmat,
        "TRASLATION": tras,
    }


def _data_emg(
    fid: BufferedReader,
    blocks: list[dict[Literal["Type", "Format", "Offset", "Size"], int]],
):
    """
    read EMG data sequence from a tdf file stream.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks : list[dict[Literal["Type", "Format", "Offset", "Size"], int]]
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 11)
    if len(block) == 0:
        return None
    ntracks, freq, time0, nframes = struct.unpack("iifi", fid.read(16))

    # channels data
    chn_map = np.array(struct.unpack("h" * ntracks, fid.read(2 * ntracks)))

    # get the data
    if block["Format"] in [1]:
        tracks = _read_tracks(fid, nframes, ntracks, 1, True)
    elif block["Format"] in [2]:
        tracks = _read_frames(fid, nframes, ntracks, 1, True)
    else:
        msg = f"block['Format'] must be 1, 2, but {block['Format']}"
        msg += " was found."
        raise ValueError(msg)

    return {
        "TRACKS": tracks,
        "CHANNELS": chn_map,
        "FREQUENCY": freq,
        "TIME0": time0,
    }


def _data_generic(
    fid: BufferedReader,
    blocks: list[dict[Literal["Type", "Format", "Offset", "Size"], int]],
):
    """
    read generic data sequence from a tdf file stream.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks : list[dict[Literal["Type", "Format", "Offset", "Size"], int]]
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 14)
    if len(block) == 0:
        return None
    ntracks, freq, time0, nframes = struct.unpack("iifi", fid.read(16))

    # channels data
    chn_map = np.array(struct.unpack("h" * ntracks, fid.read(2 * ntracks)))

    # get the data
    if block["Format"] in [1]:
        tracks = _read_tracks(fid, nframes, ntracks, 1, True)
    elif block["Format"] in [2]:
        tracks = _read_frames(fid, nframes, ntracks, 1, True)
    else:
        msg = f"block['Format'] must be 1, 2, but {block['Format']}"
        msg += " was found."
        raise ValueError(msg)

    return {
        "TRACKS": tracks,
        "CHANNELS": chn_map,
        "FREQUENCY": freq,
        "TIME0": time0,
    }


def _data_platforms(
    fid: BufferedReader,
    blocks: list[dict[Literal["Type", "Format", "Offset", "Size"], int]],
):
    """
    read generic (untracked) platforms data sequence from a tdf file stream.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks : list[dict[Literal["Type", "Format", "Offset", "Size"], int]]
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 9)
    if len(block) == 0:
        return None
    ntracks, freq, time0, nframes = struct.unpack("iifi", fid.read(16))

    # channels data
    chn_map = np.array(struct.unpack("h" * ntracks, fid.read(2 * ntracks)))
    if block["Format"] in [1, 2, 3, 4]:
        nchns = 6
    elif block["Format"] in [5, 6, 7, 8]:
        nchns = 12
    else:
        msg = f"block['Format'] must be 1, 2, but {block['Format']}"
        msg += " was found."
        raise ValueError(msg)

    # has labels
    haslbls = block["Format"] in [3, 4, 7, 8]

    # get the data
    if block["Format"] in [1, 3, 5, 7]:
        tracks = _read_tracks(fid, nframes, ntracks, nchns, haslbls)
    else:  # i.e. block["Format"] in [2, 4, 6, 8]:
        # get the labels
        if haslbls:
            lbls = []
            for trk in np.arange(ntracks):
                lbls += ["".join(struct.unpack("256B", fid.read(256))).strip()]
        else:
            lbls = [f"track{trk + 1}" for trk in np.arange(ntracks)]

        # get the available data
        data = np.nan * np.ones(
            nframes,
            ntracks,
        )
        nsamp = nchannels * ntracks * nframes
        data = struct.unpack(f"{nsamp}f", fid.read(4 * nsamp))
        data = np.reshape(data, (ntracks, nframes, nchannels))

    # return
    out: dict[str, np.ndarray[Any, np.dtype[np.float_]]] = dict(zip(lbls, data))
    return out

    return {
        "TRACKS": tracks,
        "CHANNELS": chn_map,
        "FREQUENCY": freq,
        "TIME0": time0,
    }


def read_tdf(
    path: str,
):
    """
    Return the readings from a .tdf file as dicts of 3D objects.

    Parameters
    ----------
    path: str
        an existing tdf path.

    Returns
    -------
    a dict containing the distinct data properly arranged by type.
    """

    # private signature
    tdf_signature = "41604B82CA8411D3ACB60060080C6816"

    # check the validity of the entered path
    assert os.path.exists(path), path + " does not exist."
    assert path[-4:] == ".tdf", path + ' must be an ".tdf" path.'

    # get the tdf

    # check the available data
    points = {}
    links = np.atleast_2d([])
    forceplatforms = {}
    emgchannels = {}
    imus = {}
    fid = open(path, "rb")
    try:
        # check the signature
        sig = struct.unpack("IIII", fid.read(16))
        sig = "".join([f"{b:08x}" for b in sig])
        if sig != tdf_signature.lower():
            raise IOError("invalid file")

        # get the number of entries
        _, n_entries = struct.unpack("Ii", fid.read(8))
        assert n_entries > 0, "The file specified contains no data."

        # reference indices
        ids = {
            "Point3D": 5,
            "ForcePlatform3D": 12,
            "EmgChannel": 11,
            "IMU": 17,
        }

        # check each entry to find the available blocks
        next_entry_offset = 40
        blocks = []
        for _ in range(n_entries):
            if -1 == fid.seek(next_entry_offset, 1):
                raise IOError("Error: the file specified is corrupted.")

            # get the data types
            block_info = struct.unpack("IIii", fid.read(16))
            block_labels = ["Type", "Format", "Offset", "Size"]
            block_index = dict(zip(block_labels, block_info))

            # retain only valid block types
            if block_index["Type"] in list(ids.values()):
                blocks += [block_index]

            # update the offset
            next_entry_offset = 272

        # read the available data
        for block in blocks:
            if block["Type"] == ids["Point3D"]:
                points, links = _read_point3d(fid, block)
            elif block["Type"] == ids["ForcePlatform3D"]:
                forceplatforms = _read_force3d(fid, block)
            elif block["Type"] == ids["EmgChannel"]:
                emgchannels = _read_emg(fid, block)
            elif block["Type"] == ids["EmgChannel"]:
                imus = _data_imu(fid, block)

    finally:
        fid.close()

    return {
        "point3d": points,
        "link": links,
        "force3d": forceplatforms,
        "emg": emgchannels,
        "imu": imus,
    }

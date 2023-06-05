"""btsbioengineering files reading module"""


__all__ = ["read_tdf"]


#! IMPORTS


import os
import struct
from io import BufferedReader
from typing import Any

import numpy as np
import pandas as pd


#! CONSTANTS


_BLOCK_KEYS = ["Type", "Format", "Offset", "Size"]


#! FUNCTIONS


def _get_label(
    obj: bytes,
):
    """
    _get_label convert a bytes string into a readable string

    Parameters
    ----------
    obj : bytes
        the bytes string to te read

    Returns
    -------
    label: str
        the decoded label
    """
    lbl = "".join([chr(i) for i in struct.unpack("B" * len(obj), obj)])
    return lbl.replace(chr(0), " ").strip()


def _get_block(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
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
    if len(block) > 0:
        block = block[0]
        fid.seek(block["Offset"], 0)
    else:
        block = {}
    return fid, block


def _read_frames_rts(
    fid: BufferedReader,
    nframes: int,
    cams: list[int],
):
    """
    read frames from 2D data according to the RTS (real time stream) format.

    Parameters
    ----------
    fid : BufferedReader
        file stream

    nframes : int
        number of available frames

    cams : list[int]
        the available cams

    Returns
    -------
    frames: ndarray
        the features list with shape (nframes, ncams, nfeats, 2)
    """
    frames = []
    ncams = len(cams)
    max_feats = 0

    # get the features
    for _ in np.arange(nframes):
        frame = []
        for _ in np.arange(ncams):
            nfeats = np.array(struct.unpack("i", fid.read(4)))[0]
            fid.seek(4, 1)
            vals = struct.unpack("f" * 2 * nfeats, fid.read(8 * nfeats))
            vals = np.reshape(vals, (2, nfeats), "F").T
            max_feats = max(max_feats, vals.shape[0])
            frame += [np.reshape(vals, (2, nfeats), "F").T]
        frames += [frame]

    # arrange as numpy array
    feats = np.ones((nframes, ncams, max_feats, 2)) * np.nan
    for frm, frame in enumerate(frames):
        for cam, feat in enumerate(frame):
            feats[frm, cam, np.arange(feat.shape[0]), :] = feat

    return feats


def _read_frames_pck(
    fid: BufferedReader,
    nframes: int,
    cams: list[int],
):
    """
    read frames from 2D data according to the PCK (packed data) format.

    Parameters
    ----------
    fid : BufferedReader
        file stream

    nframes : int
        number of available frames

    cams : list[int]
        the available cams

    Returns
    -------
    frames: ndarray
        the features list with shape (nframes, ncams, nfeats, 2)
    """
    ncams = len(cams)
    nsamp = int(ncams * nframes)
    nfeats = struct.unpack(f"{nsamp}h", fid.read(2 * nsamp))
    nfeats = np.reshape(nfeats, (ncams, nframes), "F")
    max_feats = int(np.max(nfeats))
    feats = np.ones((nframes, ncams, max_feats, 2)) * np.nan
    for frm in np.arange(nframes):
        for cam in np.arange(ncams):
            num = int(2 * nfeats[cam, frm])
            vals = struct.unpack(f"{num}f", fid.read(4 * num))
            vals = np.reshape(vals, (2, nfeats[cam, frm]), "F").T
            feats[frm, cam, np.arange(vals.shape[0]), :] = vals
    return feats


def _read_frames_syn(
    fid: BufferedReader,
    nframes: int,
    cams: list[int],
):
    """
    read frames from 2D data according to the SYNC (synchronized data) format.

    Parameters
    ----------
    fid : BufferedReader
        file stream

    nframes : int
        number of available frames

    cams : list[int]
        the available cams

    Returns
    -------
    frames: ndarray
        the features list with shape (nframes, ncams, nfeats, 2)
    """
    ncams = len(cams)
    max_feats = np.array(struct.unpack("1h", fid.read(2)))[0]
    shape = (nframes, ncams)
    nsamp = int(np.prod(shape))
    nfeats = struct.unpack(f"{nsamp}h", fid.read(2 * nsamp))
    nfeats = np.reshape(nfeats, shape, "F")
    feats = np.ones((nframes, ncams, max_feats, 2)) * np.nan
    for frm in np.arange(nframes):
        for cam in np.arange(ncams):
            nsamp = 2 * max_feats
            vals = struct.unpack(f"{nsamp}f", fid.read(4 * nsamp))
            vals = np.reshape(vals, (2, max_feats), "F")
            vals = vals[:, : nfeats[cam, frm]].T
            feats[frm, cam, np.arange(vals.shape[0]), :] = vals
    return feats


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
    frames: ndarray
        the features list with shape (nframes, ncams, nfeats, 2)
    """
    obj = np.ones((ntracks, nframes, nchannels)) * np.nan
    lbls = []
    for trk in np.arange(ntracks):
        # get the labels
        if haslabels:
            lbls += [_get_label(fid.read(256))]
        else:
            lbls += [f"track{trk + 1}"]

        # get the available segments
        nseg = np.array(struct.unpack("i", fid.read(4)))[0]
        fid.seek(4, 1)
        shape = (2, nseg)
        nsamp = int(np.prod(shape))
        segments = struct.unpack(f"{nsamp}i", fid.read(4 * nsamp))
        segments = np.reshape(segments, shape, "F").T

        # read the data for the actual track
        for start, stop in segments:
            for frm in np.arange(stop) + start:
                vals = fid.read(4 * nchannels)
                if frm < obj.shape[1]:
                    obj[trk, frm, :] = struct.unpack("f" * nchannels, vals)

    # split data by track
    return dict(zip(lbls, obj))


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
            label = _get_label(fid.read(256))
        else:
            label = f"track{trk + 1}"
        lbls += [label]

    # get the available data
    nsamp = nchannels * ntracks * nframes
    data = struct.unpack(f"{nsamp}f", fid.read(4 * nsamp))
    data = np.reshape(data, (ntracks, nframes, nchannels))

    # return
    return dict(zip(lbls, data))


def _read_camera_calibration(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read calibration data for general purpose

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks: list[dict[str, int]],
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
    cam_r = np.reshape(struct.unpack("9f", fid.read(36)), (3, 3), "F")  # rot mat
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
    for _ in np.arange(cam_n):
        if 1 == block["Format"]:  # Seelab type 1 calibration
            params = {
                "R": np.reshape(struct.unpack("9d", fid.read(72)), (3, 3), "F"),
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
                "R": np.reshape(struct.unpack("9d", fid.read(72)), (3, 3), "F"),
                "T": np.array(struct.unpack("3d", fid.read(24))),
                "F": np.array(struct.unpack("1d", fid.read(16))),
                "C": np.array(struct.unpack("2d", fid.read(16))),
                "KX": np.array(struct.unpack("70d", fid.read(560))),
                "KY": np.array(struct.unpack("70d", fid.read(560))),
                "VP": np.reshape(struct.unpack("4i", fid.read(16)), (2, 2), "F"),
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


def _read_data2d_calibration(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read 2D data sequence from a tdf file stream to be used
    for camera calibration.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks: list[dict[str, int]],
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
    cam_map = list(struct.unpack(f"{ncams}h", fid.read(2 * ncams)))

    # features extraction function
    if 1 == block["Format"]:  # RTS: Real Time Stream
        read_frames = _read_frames_rts
    elif 2 == block["Format"]:  # PCK: Packed Data format
        read_frames = _read_frames_pck
    elif 3 == block["Format"]:  # SYNC: Synchronized Data format
        read_frames = _read_frames_syn
    else:
        msg = f"block['Format'] must be 1, 2 or 3, but {block['Format']}"
        msg += " was found."
        raise ValueError(msg)

    # get the features
    axesfeats = read_frames(fid, naxesframes, cam_map)
    wandfeats = read_frames(fid, nwandframes, cam_map)

    return {
        "AXES": {"FEATURES": axesfeats, "PARAMS": axespars},
        "WAND": {"FEATURES": wandfeats, "PARAMS": wandpars},
        "CAMERA_CHANNELS": cam_map,
        "FREQUENCY": freq,
    }


def _read_data2d(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
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
    data: pd.DataFrame
        the available data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 4)
    if len(block) == 0:
        return None
    ncams, nframes, freq, time0 = struct.unpack("iiif", fid.read(16))
    fid.seek(4, 1)

    # channels map
    cam_map = list(struct.unpack(f"{ncams}h", fid.read(2 * ncams)))

    # features extraction
    if 1 == block["Format"]:  # RTS: Real Time Stream
        read_frames = _read_frames_rts
    elif 2 == block["Format"]:  # PCK: Packed Data format
        read_frames = _read_frames_pck
    elif 3 == block["Format"]:  # SYNC: Synchronized Data format
        read_frames = _read_frames_syn
    else:
        msg = f"block['Format'] must be 1, 2 or 3, but {block['Format']}"
        msg += " was found."
        raise ValueError(msg)

    # get the features
    return {
        "FEATURES": read_frames(fid, nframes, cam_map),
        "FREQUENCY": freq,
        "TIME0": time0,
        "CAMERA_CHANNELS": cam_map,
    }


def _read_data3d(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
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
    rmat = np.reshape(struct.unpack("9f", fid.read(36)), (3, 3), "F")
    tras = np.array(struct.unpack("3f", fid.read(12)))
    fid.seek(4, 1)

    # get links
    if block["Format"] in [1, 3]:
        nlinks = struct.unpack("i", fid.read(4))[0]
        fid.seek(4, 1)
        nsamp = 2 * nlinks
        links = struct.unpack(f"{nsamp}i", fid.read(nsamp * 4))
        links = np.reshape(links, (2, nlinks), "F").T
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

    # convert the tracks in pandas dataframes
    idx = np.arange(list(tracks.values())[0].shape[0]) / freq + time0
    idx = pd.Index(idx, name="TIME [s]")
    col = pd.MultiIndex.from_product([["X", "Y", "Z"], ["m"]])
    tracks = {i: pd.DataFrame(v, idx, col) for i, v in tracks.items()}

    # update the links with the names of the tracks
    labels = list(tracks.keys())
    links = np.array([[labels[i] for i in j] for j in links])

    return {
        "TRACKS": tracks,
        "LINKS": links,
        "DIMENSIONS": dims,
        "ROTATION_MATRIX": rmat,
        "TRASLATION": tras,
    }


def _read_optical_configuration(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read cameras physical configuration from a tdf file stream.

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
    fid, block = _get_block(fid, blocks, 6)
    if len(block) == 0:
        return None
    nchns = struct.unpack("i", fid.read(4))[0]
    fid.seek(4, 1)

    # get the data
    cameras: dict[str, Any] = {}
    for _ in np.arange(nchns):
        logicalindex = struct.unpack("i", fid.read(4))
        fid.seek(4, 1)
        lensname = _get_label(fid.read(32))
        camtype = _get_label(fid.read(32))
        camname = _get_label(fid.read(32))
        viewport = np.reshape(struct.unpack("i" * 4, fid.read(16)), (2, 2))
        cameras[camname] = {
            "INDEX": logicalindex,
            "TYPE": camtype,
            "LENS": lensname,
            "VIEWPORT": viewport,
        }

    return cameras


def _read_platforms_params(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read platforms calibration parameters from tdf file.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks: list[dict[str, int]],
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 7)
    if len(block) == 0:
        return None
    nplats = struct.unpack("i", fid.read(4))[0]
    fid.seek(4, 1)

    # channels map
    plat_map = np.array(struct.unpack(f"{nplats}h", fid.read(2 * nplats)))

    # read data for each platform
    platforms: dict[str, Any] = {}
    for i in np.arange(nplats):
        lbl = _get_label(fid.read(256))
        size = list(struct.unpack("ff", fid.read(8)))
        pos = np.reshape(struct.unpack("12f", fid.read(48)), (3, 4), "F")
        platforms[lbl] = {"SIZE": size, "POSITION": pos, "CHANNEL": plat_map[i]}

    return platforms


def _read_platforms_calibration(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read 2D data sequence from a tdf file stream to be used
    for force platforms calibration.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks: list[dict[str, int]],
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
    cam_map = list(struct.unpack(f"{ncams}h", fid.read(2 * ncams)))
    plat_map = list(struct.unpack(f"{nplats}h", fid.read(2 * nplats)))

    # features extraction function
    if 1 == block["Format"]:  # RTS: Real Time Stream
        read_frames = _read_frames_rts
    elif 2 == block["Format"]:  # PCK: Packed Data format
        read_frames = _read_frames_pck
    elif 3 == block["Format"]:  # SYNC: Synchronized Data format
        read_frames = _read_frames_syn
    else:
        msg = f"block['Format'] must be 1, 2 or 3, but {block['Format']}"
        msg += " was found."
        raise ValueError(msg)

    # read data for each platform
    platforms = []
    for plt in plat_map:
        obj = {}
        obj["CHANNEL"] += [plt]
        obj["LABEL"] += [_get_label(fid.read(32))]
        frames = np.array(struct.unpack("i", fid.read(4)))[0]
        obj["SIZE"] += [struct.unpack("ff", fid.read(8))]
        obj["FEATURES"] += [read_frames(fid, frames, cam_map)]
        platforms += [obj]

    return {
        "PLATFORMS": platforms,
        "FREQUENCY": freq,
        "CAMERA_CHANNELS": cam_map,
    }


def _read_platforms2d(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read generic (untracked) platforms data sequence from a tdf file stream.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks: list[dict[str, int]],
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
        msg = "block['Format'] must be a number in the 1-8 range, "
        msg += " was found."
        raise ValueError(msg)

    # has labels
    haslbls = block["Format"] in [3, 4, 7, 8]

    # get the data
    if block["Format"] in [1, 3, 5, 7]:
        tracks = _read_tracks(fid, nframes, ntracks, nchns, haslbls)
    else:  # i.e. block["Format"] in [2, 4, 6, 8]:
        # get the labels
        lbl = []
        for idx in np.arange(ntracks):
            if haslbls:
                lbl += [_get_label(fid.read(256))]
            else:
                lbl = [f"track{idx + 1}"]

        # get the available data
        nsamp = nchns * ntracks * nframes
        obj = struct.unpack(f"{nsamp}f", fid.read(4 * nsamp))
        obj = np.reshape(obj, (nframes, ntracks, nchns), "F")
        obj = np.transpose(obj, axes=[1, 0, 2])
        tracks = dict(zip(lbl, obj))

    # convert the tracks in a single pandas dataframe
    labels = ["ORIGIN.X", "ORIGIN.Y", "FORCE.X", "FORCE.Y", "FORCE.Z", "TORQUE"]
    units = ["m", "m", "N", "N", "N", "Nm"]
    if block["Format"] in [5, 6, 7, 8]:
        labels = ["R." + i for i in labels] + ["L." + i for i in labels]
        units += units
    cols = pd.MultiIndex.from_tuples([(i, j) for i, j in zip(labels, units)])
    for trk, obj in tracks.items():
        idx = pd.Index(np.arange(obj.shape[0]) / freq + time0, name="TIME [s]")
        tracks[trk] = pd.DataFrame(obj, index=idx, columns=cols)

    return {
        "TRACKS": tracks,
        "CHANNELS": chn_map,
    }


def _read_emg(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read EMG data sequence from a tdf file stream.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks: list[dict[str, int]],
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

    # convert the tracks in a single pandas dataframe
    tracks = pd.DataFrame({i: v.flatten() for i, v in tracks.items()})
    col = pd.MultiIndex.from_product([tracks.columns.to_numpy(), ["V"]])
    tracks.columns = col
    idx = pd.Index(np.arange(tracks.shape[0]) / freq + time0, name="TIME [s]")
    tracks.index = idx

    return {
        "TRACKS": tracks,
        "EMG_CHANNELS": chn_map,
    }


def _read_platforms3d(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read force 3D data sequence from a tdf file stream.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks: list[dict[str, int]],
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 12)
    if len(block) == 0:
        return None
    ntracks, freq, time0, nframes = struct.unpack("iifi", fid.read(16))

    # get the calibration data
    cald = np.array(struct.unpack("f" * 3, fid.read(12)))
    calr = np.reshape(struct.unpack("f" * 9, fid.read(36)), (3, 3), "F")
    calt = np.array(struct.unpack("f" * 3, fid.read(12)))
    fid.seek(4, 1)

    # get the data
    if block["Format"] in [1]:
        tracks = _read_tracks(fid, nframes, ntracks, 9, True)
    elif block["Format"] in [2]:
        tracks = _read_frames(fid, nframes, ntracks, 9, True)
    else:
        msg = f"block['Format'] must be 1, or 2, but {block['Format']}"
        msg += " was found."
        raise ValueError(msg)

    # prepare the output data
    out = {
        "TRACKS": {},
        "DIMENSIONS": cald,
        "ROTATION_MATRIX": calr,
        "TRASLATION": calt,
    }

    # convert the tracks in pandas dataframes
    tarr = np.arange(list(tracks.values())[0].shape[0]) / freq + time0
    tarr = pd.Index(tarr, name="TIME [s]")
    axs = ["X", "Y", "Z"]
    pairs = tuple(zip(["ORIGIN", "FORCE", "TORQUE"], ["m", "N", "Nm"]))
    for trk, arr in tracks.items():
        objs = {}
        for idx, pair in enumerate(pairs):
            src, unt = pair
            dims = 3 * idx + np.arange(3)
            cols = pd.MultiIndex.from_product([axs, [unt]])
            objs[src] = pd.DataFrame(arr[:, dims], index=tarr, columns=cols)
        out["TRACKS"][trk] = objs

    return out


def _read_volume(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read volumetric data sequence from a tdf file stream.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks: list[dict[str, int]],
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 13)
    if len(block) == 0:
        return None
    ntracks, freq, time0, nframes = struct.unpack("iifi", fid.read(16))

    # get the data
    if block["Format"] in [1]:  # by track
        tracks = {}
        for _ in np.arange(ntracks):
            label = _get_label(fid.read(256))

            # get the available segments
            nseg = np.array(struct.unpack("i", fid.read(4)))[0]
            fid.seek(4, 1)
            nsamp = 2 * nseg
            segments = struct.unpack(f"{nsamp}i", fid.read(4 * nsamp))
            segments = np.reshape(segments, (2, nseg), "F")

            # read the data for the actual track
            arr = np.ones((nframes, 5)) * np.nan
            for sgm in np.arange(nseg):
                for frm in np.arange(segments[0, sgm], segments[1, sgm] + 1):
                    arr[frm] = np.array(struct.unpack("ffffi", fid.read(20)))
            tracks[label] = arr

    elif block["Format"] in [2]:  # by frame
        # get the labels
        labels = []
        for _ in np.arange(ntracks):
            labels += [_get_label(fid.read(256))]

        # get the available data
        tracks = {i: np.ones((nframes, 5)) * np.nan for i in labels}
        for frm in np.arange(nframes):
            for trk in labels:
                vals = np.array(struct.unpack("ffffi", fid.read(20)))
                tracks[trk][frm, :] = vals

    else:  # errors
        msg = f"block['Format'] must be 1, 2, but {block['Format']}"
        msg += " was found."
        raise ValueError(msg)

    # convert the tracks in a single pandas dataframe
    obj = {}
    for trk, dfr in tracks.items():
        idx = pd.Index(np.arange(dfr.shape[0]) / freq + time0, name="TIME [s]")
        obj[trk] = pd.DataFrame(dfr, index=idx)

    return obj


def _read_data_generic(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read generic data sequence from a tdf file stream.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks: list[dict[str, int]],
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

    # convert the tracks in a single pandas dataframe
    tracks = pd.DataFrame(tracks)
    idx = pd.Index(np.arange(tracks.shape[0]) / freq + time0, name="TIME [s]")
    tracks.index = idx
    col = pd.MultiIndex.from_product(tracks.columns.to_numpy(), ["-"])
    tracks.columns = col
    return {
        "TRACKS": tracks,
        "CHANNELS": chn_map,
    }


def _read_calibration_generic(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read calibration data for general purpose

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks: list[dict[str, int]],
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


def _read_events(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read generic data sequence from a tdf file stream.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks: list[dict[str, int]],
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.

    time0: float
        the starting time of the event.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 16)
    if len(block) == 0:
        return None
    nevents, time0 = struct.unpack("if", fid.read(8))

    # read the events
    events: dict[str, Any] = {}
    if block["Format"] in [1]:
        for _ in np.arange(nevents):
            lbl = _get_label(fid.read(256))
            typ, nit = struct.unpack("ii", fid.read(8))
            data = struct.unpack("f" * nit, fid.read(nit * 4))
            events[lbl] = {"TYPE": typ, "DATA": data, "TIME0": time0}

    return events


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

    tdf_signature = "41604B82CA8411D3ACB60060080C6816"
    tdf: dict[str, dict[str, Any] | None] = {}
    version = float("nan")

    # check the validity of the entered path
    assert os.path.exists(path), path + " does not exist."
    assert path[-4:] == ".tdf", path + ' must be an ".tdf" path.'

    # try opening the file
    fid = open(path, "rb")
    try:
        # check the signature
        blocks = []
        next_entry_offset = 40
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

        # read all entries
        tdf["VOLUME"] = _read_volume(fid, blocks)
        tdf["DATA_CALIBRATION_GENERIC"] = _read_calibration_generic(fid, blocks)
        tdf["CAMERA_CALIBRATION"] = _read_camera_calibration(fid, blocks)
        tdf["DATA2D_CALIBRATION"] = _read_data2d_calibration(fid, blocks)
        tdf["OPTICAL_CONFIGURATION"] = _read_optical_configuration(fid, blocks)
        tdf["PLATFORMS_PARAMETERS"] = _read_platforms_params(fid, blocks)
        tdf["PLATFORMS_CALIBRATION"] = _read_platforms_calibration(fid, blocks)
        tdf["PLATFORMS2D"] = _read_platforms2d(fid, blocks)
        tdf["DATA2D"] = _read_data2d(fid, blocks)
        tdf["DATA3D"] = _read_data3d(fid, blocks)
        tdf["DATA_GENERIC"] = _read_data_generic(fid, blocks)
        tdf["EMG"] = _read_emg(fid, blocks)
        tdf["PLATFORMS3D"] = _read_platforms3d(fid, blocks)
        tdf["EVENTS"] = _read_events(fid, blocks)

    except Exception as exc:
        raise RuntimeError(exc) from exc

    finally:
        fid.close()

    return tdf  # , version

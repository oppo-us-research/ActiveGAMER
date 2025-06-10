"""
MIT License

Copyright (c) 2024 OPPO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import mmengine

from src.utils.general_utils import InfoPrinter
from tensorboardX import SummaryWriter


def init_SLAM_model(main_cfg: mmengine.Config, info_printer: InfoPrinter, logger: SummaryWriter = None):
    """initialize SLAM model

    Args:
        main_cfg     : Configuration
        info_printer : information printer
        logger       : tensorbard writer

    Returns:
        slam (SlamModel): SLAM module

    """
    ##################################################
    ### Co-SLAM
    ##################################################
    if main_cfg.slam.method == "coslam":
        info_printer("Initialize Co-SLAM...", 0, "Co-SLAM")
        from src.slam.coslam.coslam import CoSLAMNaruto as CoSLAM
        slam = CoSLAM(main_cfg, info_printer)
    elif main_cfg.slam.method == "splatam":
        info_printer("Initialize SplaTAM...", 0, "SplaTAM")
        from src.slam.splatam.splatam import SplatamOurs as SplaTAM
        slam = SplaTAM(main_cfg, info_printer, logger)
    else:
            assert False, f"SLAM choices: [coslam, splatam]. Current option: [{main_cfg.slam.method}]"
    return slam
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

from icon4pytools.common.logger import setup_logger
from icon4pytools.liskov.codegen.integration.interface import (
    EndDeleteData,
    EndFusedStencilData,
    EndStencilData,
    IntegrationCodeInterface,
    StartDeleteData,
    StartFusedStencilData,
    StartStencilData,
    UnusedDirective,
)
from icon4pytools.liskov.codegen.shared.types import CodeGenInput
from icon4pytools.liskov.pipeline.definition import Step


logger = setup_logger(__name__)


class StencilTransformer(Step):
    def __init__(self, parsed: IntegrationCodeInterface, fused: bool) -> None:
        self.parsed = parsed
        self.fused = fused

    def __call__(self, data: Any = None) -> IntegrationCodeInterface:
        """Transform stencils in the parse tree based on the 'fused' flag, transforming or removing as necessary.

        This method processes stencils present in the 'parsed' object according to the 'fused'
        flag. If 'fused' is True, it identifies and processes stencils that are eligible for
        deletion. If 'fused' is False, it removes fused stencils.

        Args:
            data (Any): Optional data to be passed. Default is None.

        Returns:
            IntegrationCodeInterface: The interface object along with any transformations applied.
        """
        if self.fused:
            logger.info("Transforming stencils for deletion.")
            self._process_stencils_for_deletion()
        else:
            logger.info("Removing fused stencils.")
            self._remove_fused_stencils()
            self._remove_delete()

        return self.parsed

    def _process_stencils_for_deletion(self) -> None:
        stencils_to_remove = []

        for start_fused, end_fused in zip(
            self.parsed.StartFusedStencil, self.parsed.EndFusedStencil, strict=True
        ):
            for start_single, end_single in zip(
                self.parsed.StartStencil, self.parsed.EndStencil, strict=True
            ):
                if self._stencil_is_removable(start_fused, end_fused, start_single, end_single):
                    self._create_delete_directives(start_single, end_single)
                    stencils_to_remove += [start_single, end_single]

        self._remove_stencils(stencils_to_remove)

    def _stencil_is_removable(
        self,
        start_fused: StartFusedStencilData,
        end_fused: EndFusedStencilData,
        start_single: StartStencilData,
        end_single: EndStencilData,
    ) -> bool:
        return (
            start_fused.startln < start_single.startln
            and start_single.startln < end_fused.startln
            and start_fused.startln < end_single.startln
            and end_single.startln < end_fused.startln
        )

    def _create_delete_directives(
        self, start_single: StartStencilData, end_single: EndStencilData
    ) -> None:
        for attr, param in zip(["StartDelete", "EndDelete"], [start_single, end_single]):
            directive = getattr(self.parsed, attr)
            if directive == UnusedDirective:
                directive = []

            if attr == "StartDelete":
                cls = StartDeleteData
            elif attr == "EndDelete":
                cls = EndDeleteData

            directive.append(cls(startln=param.startln))
            setattr(self.parsed, attr, directive)

    def _remove_stencils(self, stencils_to_remove: list[CodeGenInput]) -> None:
        attributes_to_modify = ["StartStencil", "EndStencil"]

        for attr_name in attributes_to_modify:
            current_stencil_list = getattr(self.parsed, attr_name)
            modified_stencil_list = [_ for _ in current_stencil_list if _ not in stencils_to_remove]
            setattr(self.parsed, attr_name, modified_stencil_list)

    def _remove_fused_stencils(self) -> None:
        self.parsed.StartFusedStencil = []
        self.parsed.EndFusedStencil = []

    def _remove_delete(self) -> None:
        self.parsed.StartDelete = []
        self.parsed.EndDelete = []

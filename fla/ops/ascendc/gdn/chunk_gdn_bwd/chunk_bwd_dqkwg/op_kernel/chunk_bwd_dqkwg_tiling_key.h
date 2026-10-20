/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_BWD_DQKWG_TILING_KEY_H
#define CHUNK_BWD_DQKWG_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// 单轴：false 为安全 P1/u32，true 保持原 wide/key1 语义。
ASCENDC_TPL_ARGS_DECL(ChunkBwdDqkwg,
    ASCENDC_TPL_BOOL_DECL(WIDE_SCHEDULE, 0, 1)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(WIDE_SCHEDULE, 0)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(WIDE_SCHEDULE, 1))
);

#endif // CHUNK_BWD_DQKWG_TILING_KEY_H

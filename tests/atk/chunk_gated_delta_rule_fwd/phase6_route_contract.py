#!/usr/bin/env python3
"""编译真实 L2 分流函数，验证可选输出不改变 Phase6 路由及扩展能力隔离。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
IMPLEMENTATION = Path(
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/"
    "op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.cpp"
)


def definition(source: str, marker: str, *, trailing: str = "") -> str:
    """截取完整定义，测试直接使用生产函数，不在 Python 中重写分流条件。"""
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1] + trailing
    raise ValueError(f"定义不完整：{marker}")


HARNESS = r'''
int main()
{
    aclTensor tensor;
    aclIntArray indices;
    int checked = 0;
    int failed = 0;
    auto check = [&](const ChunkGatedDeltaRuleFwdParams &params, bool expected,
                     const char *name) {
        ++checked;
        if (UsePreparePath(params) != expected) {
            ++failed;
            std::cerr << name << " layout=" << params.layout
                      << " chunk=" << params.chunkSize
                      << " A=" << (params.aOutOptional != nullptr) << '\n';
        }
    };
    // 合法 Phase6 的布局/块长、初态/终态、定长/变长与两个可选辅助输出。
    for (const char *layout : {"BNSD", "NTD"}) {
        for (int chunk : {64, 128}) {
            for (unsigned mask = 0; mask < 32; ++mask) {
                ChunkGatedDeltaRuleFwdParams params;
                params.layout = layout;
                params.chunkSize = chunk;
                params.aOutOptional = (mask & 1) ? &tensor : nullptr;
                params.gCumsumOutOptional = (mask & 2) ? &tensor : nullptr;
                params.initialStateOptional = (mask & 4) ? &tensor : nullptr;
                params.finalStateOutOptional = (mask & 8) ? &tensor : nullptr;
                params.cuSeqlensOptional = (mask & 16) ? &indices : nullptr;
                params.chunkIndicesOptional = params.cuSeqlensOptional;
                check(params, false, "phase6");
            }
        }
    }
    // 每种超出 Phase6 的能力独立触发 Prepare，不能因 A 为空误入 Phase6。
    for (bool outputA : {false, true}) {
        for (int feature = 0; feature < 10; ++feature) {
            ChunkGatedDeltaRuleFwdParams params;
            params.layout = "BNSD";
            params.aOutOptional = outputA ? &tensor : nullptr;
            switch (feature) {
                case 0: params.useExp2 = true; break;
                case 1: params.useQkL2norm = true; break;
                case 2: params.aLogOptional = &tensor; break;
                case 3: params.dtBiasOptional = &tensor; break;
                case 4: params.betaEffOutOptional = &tensor; break;
                case 5: params.allowNegEigval = true; break;
                case 6: params.hOutOptional = &tensor; break;
                case 7: params.stateVFirst = true; break;
                case 8: params.layout = "BSND"; break;
                case 9: params.layout = "TND"; break;
            }
            check(params, true, "prepare");
        }
    }
    std::cout << "{\"checked\":" << checked << ",\"failed\":" << failed << "}\n";
    return failed == 0 ? 0 : 1;
}
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default=os.environ.get("CXX", "c++"),
                        help="可用的 C++14 编译器，默认 CXX 或 c++")
    parser.add_argument("--source-ref", help="可选 Git 提交，用于对照修复前路由")
    parser.add_argument("--build-dir", type=Path, help="临时构建的父目录")
    args = parser.parse_args()
    compiler = shutil.which(args.compiler)
    if compiler is None:
        parser.error(f"找不到 C++ 编译器：{args.compiler}")
    if args.source_ref:
        source = subprocess.check_output(
            ["git", "show", f"{args.source_ref}:{IMPLEMENTATION.as_posix()}"],
            cwd=ROOT, text=True, encoding="utf-8",
        )
    else:
        source = (ROOT / IMPLEMENTATION).read_text(encoding="utf-8")
    struct = definition(source, "struct ChunkGatedDeltaRuleFwdParams", trailing=";")
    selector = definition(source, "static bool UsePreparePath")
    preamble = (
        "#include <cstdint>\n#include <cstring>\n#include <iostream>\n"
        "#include <initializer_list>\nstruct aclTensor {};\nstruct aclIntArray {};\n"
        "constexpr int64_t CHUNK_GATED_DELTA_RULE_FWD_CHUNK_64 = 64;\n"
    )
    with tempfile.TemporaryDirectory(prefix="gdn-route-", dir=args.build_dir) as tmp:
        directory = Path(tmp)
        cpp = directory / "route.cpp"
        executable = directory / ("route.exe" if os.name == "nt" else "route")
        cpp.write_text(preamble + struct + "\n" + selector + "\n" + HARNESS,
                       encoding="utf-8")
        subprocess.run([compiler, "-std=c++14", "-Wall", "-Wextra", "-Werror",
                        str(cpp), "-o", str(executable)], check=True)
        result = subprocess.run([str(executable)], capture_output=True,
                                text=True, encoding="utf-8")
        if result.stderr:
            print(result.stderr, end="")
        summary = json.loads(result.stdout)
        summary.update(result="passed" if result.returncode == 0 else "failed",
                       source_ref=args.source_ref or "working-tree")
        print(json.dumps(summary, ensure_ascii=False))
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

#ifndef FLA_CHUNK_STATE_CONTRACT_H
#define FLA_CHUNK_STATE_CONTRACT_H

#include <cstdint>
#include <cstddef>

namespace fla {
struct ChunkArrayView {
    const int64_t *data;
    size_t size;
    size_t Size() const { return size; }
    int64_t operator[](size_t index) const { return data[index]; }
};
// Host-only validation shared by ACLNN entry points. No device data reads.
template <typename IntArray>
inline bool ValidateStateChunks(const IntArray *cu, const IntArray *indices,
                                int64_t batch, int64_t tokens, int64_t chunkSize,
                                int64_t &chunks, bool allowMissingIndices = false,
                                bool allowEmptySequences = false)
{
    if (chunkSize <= 0 || batch <= 0 || tokens <= 0) return false;
    chunks = (tokens + chunkSize - 1) / chunkSize;
    if (cu == nullptr) return indices == nullptr;
    if (batch != 1 || cu->Size() < 2 || (*cu)[0] != 0 ||
        (*cu)[cu->Size() - 1] != tokens || (!allowMissingIndices && indices == nullptr)) return false;
    chunks = 0;
    size_t offset = 0;
    for (size_t seq = 0; seq + 1 < cu->Size(); ++seq) {
        const int64_t length = (*cu)[seq + 1] - (*cu)[seq];
        if (length < 0 || (!allowEmptySequences && length == 0)) return false;
        const int64_t count = (length + chunkSize - 1) / chunkSize;
        if (indices != nullptr) {
            for (int64_t chunk = 0; chunk < count; ++chunk) {
                if (offset + 1 >= indices->Size() || (*indices)[offset] != static_cast<int64_t>(seq) ||
                    (*indices)[offset + 1] != chunk) return false;
                offset += 2;
            }
        }
        chunks += count;
    }
    return chunks > 0 && (indices == nullptr || offset == indices->Size());
}
} // namespace fla
#endif

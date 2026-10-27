#include "rt/image.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <memory>
#include <utility>

namespace rt {
namespace {

struct FileCloser { void operator()(FILE* f) const { if (f) fclose(f); } };
using File = std::unique_ptr<FILE, FileCloser>;

uint32_t crc32_update(uint32_t crc, const uint8_t* data, size_t n) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : (c >> 1);
            table[i] = c;
        }
        init = true;
    }
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

void put32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back((uint8_t)(x >> 24)); v.push_back((uint8_t)(x >> 16));
    v.push_back((uint8_t)(x >> 8));  v.push_back((uint8_t)x);
}

bool write_chunk(FILE* f, const char* type, const std::vector<uint8_t>& data) {
    std::vector<uint8_t> hdr;
    put32(hdr, (uint32_t)data.size());
    if (fwrite(hdr.data(), 1, 4, f) != 4) return false;
    std::vector<uint8_t> td(type, type + 4);
    td.insert(td.end(), data.begin(), data.end());
    if (fwrite(td.data(), 1, td.size(), f) != td.size()) return false;
    std::vector<uint8_t> crc;
    put32(crc, crc32_update(0, td.data(), td.size()));
    return fwrite(crc.data(), 1, 4, f) == 4;
}

}  // namespace

bool save_png(const Image& img, const std::string& path, bool apply_srgb) {
    if (img.width <= 0 || img.height <= 0) return false;
    const size_t w = img.width, h = img.height;

    // Raw scanlines: filter byte 0 + RGB bytes.
    std::vector<uint8_t> raw(h * (1 + 3 * w));
    for (size_t y = 0; y < h; ++y) {
        uint8_t* row = &raw[y * (1 + 3 * w)];
        row[0] = 0;
        for (size_t x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) {
                float v = img.rgb[(y * w + x) * 3 + c];
                v = apply_srgb ? linear_to_srgb(v) : clampf(v, 0.f, 1.f);
                row[1 + 3 * x + c] = (uint8_t)(v * 255.f + 0.5f);
            }
    }

    // zlib stream using "stored" (uncompressed) deflate blocks: no zlib needed.
    std::vector<uint8_t> z = {0x78, 0x01};
    size_t pos = 0;
    while (pos < raw.size()) {
        size_t n = raw.size() - pos;
        if (n > 65535) n = 65535;
        bool last = (pos + n == raw.size());
        z.push_back(last ? 1 : 0);
        z.push_back((uint8_t)(n & 0xFF));        z.push_back((uint8_t)(n >> 8));
        z.push_back((uint8_t)(~n & 0xFF));       z.push_back((uint8_t)((~n >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + n);
        pos += n;
    }
    uint32_t a = 1, b = 0;
    for (uint8_t byte : raw) { a = (a + byte) % 65521; b = (b + a) % 65521; }
    put32(z, (b << 16) | a);

    File f(fopen(path.c_str(), "wb"));
    if (!f) return false;
    const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (fwrite(sig, 1, 8, f.get()) != 8) return false;
    std::vector<uint8_t> ihdr;
    put32(ihdr, (uint32_t)w); put32(ihdr, (uint32_t)h);
    ihdr.push_back(8); ihdr.push_back(2); ihdr.push_back(0); ihdr.push_back(0); ihdr.push_back(0);
    return write_chunk(f.get(), "IHDR", ihdr) && write_chunk(f.get(), "IDAT", z) &&
           write_chunk(f.get(), "IEND", {});
}

bool save_pfm(const Image& img, const std::string& path) {
    if (img.width <= 0 || img.height <= 0) return false;
    File f(fopen(path.c_str(), "wb"));
    if (!f) return false;
    fprintf(f.get(), "PF\n%d %d\n-1.0\n", img.width, img.height);   // -1.0 = little endian
    // PFM stores rows bottom-to-top.
    for (int y = img.height - 1; y >= 0; --y) {
        const float* row = &img.rgb[(size_t)y * img.width * 3];
        if (fwrite(row, sizeof(float), (size_t)img.width * 3, f.get()) != (size_t)img.width * 3) return false;
    }
    return true;
}

bool load_pfm(const std::string& path, Image& out) {
    File f(fopen(path.c_str(), "rb"));
    if (!f) return false;
    char magic[3] = {0};
    int w = 0, h = 0;
    float scale = 0.f;
    if (fscanf(f.get(), "%2s %d %d %f", magic, &w, &h, &scale) != 4) return false;
    if (strcmp(magic, "PF") != 0 || w <= 0 || h <= 0) return false;
    fgetc(f.get());   // single whitespace after the header
    Image img(w, h);
    for (int y = h - 1; y >= 0; --y) {
        float* row = &img.rgb[(size_t)y * w * 3];
        if (fread(row, sizeof(float), (size_t)w * 3, f.get()) != (size_t)w * 3) return false;
    }
    if (scale > 0.f) {   // big endian file: swap bytes
        for (float& v : img.rgb) {
            unsigned char* p = reinterpret_cast<unsigned char*>(&v);
            std::swap(p[0], p[3]); std::swap(p[1], p[2]);
        }
    }
    out = std::move(img);
    return true;
}

}  // namespace rt

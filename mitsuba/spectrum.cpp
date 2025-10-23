#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/mmap.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/spectrum.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Scalar>
void spectrum_from_file(const fs::path &path, std::vector<Scalar> &wavelengths,
                        std::vector<Scalar> &values) {

    auto fs = Thread::thread()->file_resolver();
    fs::path file_path = fs->resolve(path);
    if (!fs::exists(file_path))
        Log(Error, "\"%s\": file does not exist!", file_path);

    Log(Info, "Loading spectral data file \"%s\" ..", file_path);
    std::string extension = string::to_lower(file_path.extension().string());
    if (extension == ".spd") {
        ref<MemoryMappedFile> mmap = new MemoryMappedFile(file_path, false);
        char *current = (char *) mmap->data(),
             *end     = current + mmap->size(),
             *tmp;
        bool comment = false;
        size_t counter = 0;
        while (current != end) {
            char c = *current;
            if (c == '#') {
                comment = true;
                current++;
            } else if (c == '\n') {
                comment = false;
                counter = 0;
                current++;
            } else if (!comment && c != ' ' && c != '\r') {
                Scalar val = string::parse_float<Scalar>(current, end, &tmp);
                current = tmp;
                if (counter == 0)
                    wavelengths.push_back(val);
                else if (counter == 1)
                    values.push_back(val);
                else
                    Log(Error, "While parsing the file, more than two elements were defined in a line");
                counter++;
            } else {
                current++;
            }
        }
    } else {
        Log(Error, "You need to provide a valid extension like \".spd\" to read"
                   "the information from an ASCII file. You used \"%s\"", extension);
    }
}

template <typename Scalar>
void spectrum_to_file(const fs::path &path, const std::vector<Scalar> &wavelengths,
                      const std::vector<Scalar> &values) {

    auto fs = Thread::thread()->file_resolver();
    fs::path file_path = fs->resolve(path);

    if (wavelengths.size() != values.size())
        Log(Error, "Wavelengths size (%u) need to be equal to values size (%u)",
            wavelengths.size(), values.size());

    Log(Info, "Writing spectral data to file \"%s\" ..", file_path);
    ref<FileStream> file = new FileStream(file_path, FileStream::ETruncReadWrite);
    std::string extension = string::to_lower(file_path.extension().string());

    if (extension == ".spd") {
        // Write file with textual spectra format
        for (size_t i = 0; i < wavelengths.size(); ++i) {
            std::ostringstream oss;
            oss << wavelengths[i] << " " << values[i];
            file->write_line(oss.str());
        }
    } else {
        Log(Error, "You need to provide a valid extension like \".spd\" to store"
                   "the information in an ASCII file. You used \"%s\"", extension);
    }
}

template <typename Scalar>
Color<Scalar, 3> spectrum_list_to_srgb(const std::vector<Scalar> &wavelengths,
                                       const std::vector<Scalar> &values,
                                       bool bounded, bool d65) {
    Color<Scalar, 3> xyz = (Scalar) 0.f;

    const int steps = 1000;
    for (int i = 0; i < steps; ++i) {
        Scalar w = (Scalar) MI_CIE_MIN +
                   (i / (Scalar)(steps - 1)) * ((Scalar) MI_CIE_MAX -
                                                (Scalar) MI_CIE_MIN);

        if (w < wavelengths.front() || w > wavelengths.back())
            continue;

        // Find interval containing 'x'
        uint32_t index = math::find_interval<uint32_t>(
            (uint32_t) wavelengths.size(),
            [&](uint32_t idx) {
                return wavelengths[idx] <= w;
            });

        Scalar w0 = wavelengths[index],
               w1 = wavelengths[index + 1],
               v0 = values[index],
               v1 = values[index + 1];

        // Linear interpolant at 'w'
        Scalar v = ((w - w1) * v0 + (w0 - w) * v1) / (w0 - w1);
        xyz += cie1931_xyz(w) * v * (d65 ? cie_d65(w) : Scalar(1));
    }

    // Last specified value repeats implicitly
    xyz *= ((Scalar) MI_CIE_MAX - (Scalar) MI_CIE_MIN) / (Scalar) steps;
    Color<Scalar, 3> rgb = xyz_to_srgb(xyz);

    if (bounded && dr::any(rgb < (Scalar) 0.f || rgb > (Scalar) 1.f)) {
        Log(Warn, "Spectrum: clamping out-of-gamut color %s", rgb);
        rgb = clip(rgb, (Scalar) 0.f, (Scalar) 1.f);
    } else if (!bounded && dr::any(rgb < (Scalar) 0.f)) {
        Log(Warn, "Spectrum: clamping out-of-gamut color %s", rgb);
        rgb = dr::maximum(rgb, (Scalar) 0.f);
    }

    return rgb;
}

/// Explicit instantiations
template MI_EXPORT_LIB void spectrum_from_file(const fs::path &path,
                                               std::vector<float> &wavelengths,
                                               std::vector<float> &values);
template MI_EXPORT_LIB void spectrum_from_file(const fs::path &path,
                                               std::vector<double> &wavelengths,
                                               std::vector<double> &values);

template MI_EXPORT_LIB void spectrum_to_file(const fs::path &path,
                                             const std::vector<float> &wavelengths,
                                             const std::vector<float> &values);
template MI_EXPORT_LIB void spectrum_to_file(const fs::path &path,
                                             const std::vector<double> &wavelengths,
                                             const std::vector<double> &values);

template MI_EXPORT_LIB Color<float, 3>
spectrum_list_to_srgb(const std::vector<float> &wavelengths,
                      const std::vector<float> &values, bool bounded, bool d65);
template MI_EXPORT_LIB Color<double, 3>
spectrum_list_to_srgb(const std::vector<double> &wavelengths,
                      const std::vector<double> &values, bool bounded, bool d65);

// =======================================================================
//! @{ \name CIE 1931 2 degree observer implementation
// =======================================================================
using Float = float;

static const Float cie1931_tbl[MI_CIE_SAMPLES * 3] = {
    Float(0.00000000000000000e+00), Float(2.57457718000000011e-06), Float(2.03611930999999999e-05), Float(6.84968610000000048e-05),
Float(1.62470436999999997e-04), Float(3.17165400999999973e-04), Float(5.47873991999999984e-04), Float(8.70154914999999966e-04),
Float(1.29866658000000004e-03), Float(1.84812688000000002e-03), Float(2.50663611000000001e-03), Float(3.23017681000000020e-03),
Float(3.97336898000000011e-03), Float(4.69039023999999992e-03), Float(5.33591490999999959e-03), Float(5.86375584000000012e-03),
Float(6.22856407000000039e-03), Float(6.38534767000000034e-03), Float(6.29656550000000023e-03), Float(5.98657494000000030e-03),
Float(5.50088533000000005e-03), Float(4.88410055000000013e-03), Float(4.18288894999999973e-03), Float(3.44251410999999981e-03),
Float(2.70870527999999980e-03), Float(2.02699768999999985e-03), Float(1.44412274999999998e-03), Float(1.10229318000000000e-03),
Float(1.30560647999999999e-03), Float(2.37887906000000017e-03), Float(4.64806258000000009e-03), Float(8.43001037000000059e-03),
Float(1.40552970000000004e-02), Float(2.18464438000000001e-02), Float(3.21179763000000013e-02), Float(4.52146549000000030e-02),
Float(6.15170220999999967e-02), Float(8.14383885000000012e-02), Float(1.05429258000000000e-01), Float(1.33890770000000006e-01),
Float(1.67238348000000001e-01), Float(2.05937662000000002e-01), Float(2.50370106000000013e-01), Float(3.00960748000000018e-01),
Float(3.57058096000000022e-01), Float(4.15080047000000013e-01), Float(4.71023920000000025e-01), Float(5.20836967000000026e-01),
Float(5.60541402000000043e-01), Float(5.86086528999999974e-01), Float(5.93415080000000041e-01), Float(5.78601807000000032e-01),
Float(5.37892854000000037e-01), Float(4.73720440000000018e-01), Float(3.93547511000000008e-01), Float(3.04886079999999977e-01),
Float(2.15361781999999992e-01), Float(1.32509361000000007e-01), Float(6.39948227999999984e-02), Float(1.73252771999999990e-02),
Float(6.14995524000000032e-17),

    Float(0.00000000000000000e+00), Float(5.90110864000000013e-06), Float(4.66692601999999976e-05), Float(1.56999534000000013e-04),
Float(3.72393459000000006e-04), Float(7.26964996000000006e-04), Float(1.25576501000000001e-03), Float(1.99445513999999989e-03),
Float(2.97663345000000017e-03), Float(4.24636189999999959e-03), Float(6.11507771999999974e-03), Float(9.20780856000000085e-03),
Float(1.41804137000000000e-02), Float(2.16828633000000007e-02), Float(3.23531801999999979e-02), Float(4.68611535999999967e-02),
Float(6.58451377999999948e-02), Float(8.99408418000000043e-02), Float(1.19571698000000004e-01), Float(1.53495440000000011e-01),
Float(1.89896898000000008e-01), Float(2.26956493000000009e-01), Float(2.62844281999999985e-01), Float(2.95745244000000018e-01),
Float(3.23805114000000005e-01), Float(3.45234099999999988e-01), Float(3.58211932999999982e-01), Float(3.61928876000000010e-01),
Float(3.57472387000000003e-01), Float(3.46039935999999992e-01), Float(3.28856522000000010e-01), Float(3.07170686999999978e-01),
Float(2.82190134999999987e-01), Float(2.55150073999999987e-01), Float(2.27278602999999989e-01), Float(1.99741015000000013e-01),
Float(1.73121080000000003e-01), Float(1.47666047000000001e-01), Float(1.23631978999999995e-01), Float(1.01257948000000001e-01),
Float(8.07870656999999945e-02), Float(6.24831631999999991e-02), Float(4.65759452999999980e-02), Float(3.33130872999999990e-02),
Float(2.28127353999999994e-02), Float(1.47846046000000007e-02), Float(8.90185399000000034e-03), Float(4.83611019000000014e-03),
Float(2.24424404000000012e-03), Float(7.98052438999999990e-04), Float(1.63543477000000004e-04), Float(4.56862343999999982e-06),
Float(0.00000000000000000e+00), Float(0.00000000000000000e+00), Float(0.00000000000000000e+00), Float(0.00000000000000000e+00),
Float(0.00000000000000000e+00), Float(0.00000000000000000e+00), Float(0.00000000000000000e+00), Float(0.00000000000000000e+00),
Float(0.00000000000000000e+00),

    Float(0.00000000000000000e+00), Float(4.24179287999999988e-01), Float(7.17662376000000046e-01), Float(8.98671772000000035e-01),
Float(9.85180931000000037e-01), Float(9.96288276000000028e-01), Float(9.50235779999999974e-01), Float(8.65420039999999946e-01),
Float(7.60496873000000018e-01), Float(6.53599060999999981e-01), Float(5.54951743000000053e-01), Float(4.65013078999999996e-01),
Float(3.83724016999999973e-01), Float(3.10975975999999987e-01), Float(2.46642019999999990e-01), Float(1.90671067999999999e-01),
Float(1.42936132999999993e-01), Float(1.03320711999999995e-01), Float(7.16913822000000006e-02), Float(4.72741080000000022e-02),
Float(2.91314046000000007e-02), Float(1.63691343999999989e-02), Float(8.00842868000000048e-03), Float(3.12345548000000006e-03),
Float(7.90931012999999982e-04), Float(5.82659945000000025e-05), Float(0.00000000000000000e+00), Float(0.00000000000000000e+00),
Float(0.00000000000000000e+00), Float(0.00000000000000000e+00), Float(0.00000000000000000e+00), Float(0.00000000000000000e+00),
Float(0.00000000000000000e+00), Float(0.00000000000000000e+00), Float(0.00000000000000000e+00), Float(0.00000000000000000e+00),
Float(0.00000000000000000e+00), Float(0.00000000000000000e+00), Float(0.00000000000000000e+00), Float(0.00000000000000000e+00),
Float(0.00000000000000000e+00), Float(0.00000000000000000e+00), Float(0.00000000000000000e+00), Float(0.00000000000000000e+00),
Float(0.00000000000000000e+00), Float(0.00000000000000000e+00), Float(0.00000000000000000e+00), Float(0.00000000000000000e+00),
Float(0.00000000000000000e+00), Float(0.00000000000000000e+00), Float(0.00000000000000000e+00), Float(0.00000000000000000e+00),
Float(0.00000000000000000e+00), Float(0.00000000000000000e+00), Float(0.00000000000000000e+00), Float(0.00000000000000000e+00),
Float(0.00000000000000000e+00)

};


NAMESPACE_BEGIN(detail)
CIE1932Tables<float> color_space_tables_scalar;
#if defined(MI_ENABLE_LLVM)
CIE1932Tables<dr::LLVMArray<float>> color_space_tables_llvm;
#endif
#if defined(MI_ENABLE_CUDA)
CIE1932Tables<dr::CUDAArray<float>> color_space_tables_cuda;
#endif
NAMESPACE_END(detail)

void color_management_static_initialization(bool cuda, bool llvm) {
    detail::color_space_tables_scalar.initialize(cie1931_tbl);
#if defined(MI_ENABLE_LLVM)
    if (llvm)
        detail::color_space_tables_llvm.initialize(cie1931_tbl);
#endif
#if defined(MI_ENABLE_CUDA)
    if (cuda)
        detail::color_space_tables_cuda.initialize(cie1931_tbl);
#endif
    (void) cuda; (void) llvm;
}

void color_management_static_shutdown() {
    detail::color_space_tables_scalar.release();
#if defined(MI_ENABLE_LLVM)
    detail::color_space_tables_llvm.release();
#endif
#if defined(MI_ENABLE_CUDA)
    detail::color_space_tables_cuda.release();
#endif
}

//! @}
// =======================================================================

NAMESPACE_END(mitsuba)

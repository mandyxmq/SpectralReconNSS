import sys
sys.path.insert(0, '/Users/mengqixia/Github/mitsuba3/build/python')
import drjit as dr
import mitsuba as mi

class spectralDiffuse(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)

        self.reflectance = props['reflectance']
        
        self.m_flags = mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide
        self.m_components = [mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide]

        self.wavelength_num = 50
        self.spectrum = mi.Texture1f(dr.full(mi.TensorXf, 1.0, shape=(self.wavelength_num, 1)))
        self.wavelengths = dr.full(mi.TensorXf, 0.0, shape=(self.wavelength_num, 1))

    def sample(self, ctx, si, sample1, sample2, active):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        bs = mi.BSDFSample3f()
        
        active &= (cos_theta_i > 0)
        if not (ctx.is_enabled(mi.BSDFFlags.DiffuseReflection)) and (active):
            return bs, mi.Spectrum(0.0)
        
        bs.wo = mi.warp.square_to_cosine_hemisphere(sample2)
        bs.pdf = mi.warp.square_to_cosine_hemisphere_pdf(bs.wo)
        bs.eta = 1.0
        bs.sampled_type = mi.Float(mi.BSDFFlags.DiffuseReflection)
        bs.sampled_component = 0
        
        # Bspline implementation
        ratio = (si.wavelengths - 400.0) / (700.0 - 400.0)
        color = mi.unpolarized_spectrum(0.0)
        color[0] = self.spectrum.eval(mi.Float(ratio[0]))[0]
        color[1] = self.spectrum.eval(mi.Float(ratio[1]))[0]
        color[2] = self.spectrum.eval(mi.Float(ratio[2]))[0]
        color[3] = self.spectrum.eval(mi.Float(ratio[3]))[0]

        value = color

        return bs, dr.select(active & (bs.pdf > 0), mi.depolarizer(value), mi.Spectrum(0.0))
    

    def eval(self, ctx, si, wo, active):
        if not ctx.is_enabled(mi.BSDFFlags.DiffuseReflection):
            return mi.Spectrum(0.0)
        
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        
        active &= (cos_theta_i > 0) & (cos_theta_o > 0)

        # Bspline implementation
        ratio = (si.wavelengths - 400.0) / (700.0 - 400.0)
        color = mi.unpolarized_spectrum(0.0)
        color[0] = self.spectrum.eval(mi.Float(ratio[0]))[0]
        color[1] = self.spectrum.eval(mi.Float(ratio[1]))[0]
        color[2] = self.spectrum.eval(mi.Float(ratio[2]))[0]
        color[3] = self.spectrum.eval(mi.Float(ratio[3]))[0]

        value = color * dr.rcp(dr.pi) * cos_theta_o

        return dr.select(active, mi.depolarizer(value), mi.Spectrum(0.0))
    

    def pdf(self, ctx, si, wo, active):
        if not ctx.is_enabled(mi.BSDFFlags.DiffuseReflection):
            return 0.0
        
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        
        pdf_val = mi.warp.square_to_cosine_hemisphere_pdf(wo)
        return dr.select((cos_theta_i > 0) & (cos_theta_o > 0), pdf_val, 0.0)
    

    def eval_pdf(self, ctx, si, wo, active):
        if not ctx.is_enabled(mi.BSDFFlags.DiffuseReflection):
            return mi.Spectrum(0.0), 0.0
        
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        
        active &= (cos_theta_i > 0) & (cos_theta_o > 0)

        # Bspline implementation
        ratio = (si.wavelengths - 400.0) / (700.0 - 400.0)
        color = mi.unpolarized_spectrum(0.0)

        color[0] = self.spectrum.eval(mi.Float(ratio[0]))[0]
        color[1] = self.spectrum.eval(mi.Float(ratio[1]))[0]
        color[2] = self.spectrum.eval(mi.Float(ratio[2]))[0]
        color[3] = self.spectrum.eval(mi.Float(ratio[3]))[0]

        value = color * dr.rcp(dr.pi) * cos_theta_o

        pdf_val = mi.warp.square_to_cosine_hemisphere_pdf(wo)
        
        return (
            dr.select(active, mi.depolarizer(value), mi.Spectrum(0.0)),
            dr.select(active, pdf_val, 0.0)
        )
    
    def traverse(self, callback):
        callback.put_object('reflectance', self.reflectance, mi.ParamFlags.Differentiable)
        callback.put_parameter('spectrum', self.spectrum, mi.ParamFlags.Differentiable)
        callback.put_parameter('spectrum_value', self.spectrum.tensor(), mi.ParamFlags.Differentiable)
        callback.put_parameter('wavelengths', self.wavelengths, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        pass
        #self.spectrum.set_tensor(self.spectrum.tensor())

    def to_string(self):
        return ('spectralDiffuse[]')

mi.register_bsdf("spectralDiffuse", lambda props: spectralDiffuse(props))
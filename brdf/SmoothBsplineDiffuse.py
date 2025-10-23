import drjit as dr
import mitsuba as mi
from bspline import *

mi.set_variant('cuda_ad_spectral')

class SmoothBsplineDiffuse(mi.BSDF):
    def __init__(self, props):
        super().__init__(props)
        self.m_reflectance = props.get('reflectance', 0.5)
        
        if isinstance(self.m_reflectance, (float, int)):
            self.m_reflectance = mi.Texture(mi.Spectrum(self.m_reflectance))
        
        self.m_flags = mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide
        self.m_components = [self.m_flags]

        self.spline_degree = 3
        self.knots = mi.Float([1.0])
        self.coeff = mi.Float([1.0])

    def traverse(self, callback):
        callback.put_object('reflectance', self.m_reflectance, mi.ParamFlags.Differentiable)
        callback.put_parameter('coeff', self.coeff, mi.ParamFlags.Differentiable)
        callback.put_parameter('knots', self.knots, mi.ParamFlags.NonDifferentiable)

    def sample(self, ctx, si, sample1, sample2, active=True):
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
        for i in range(4):
            color[i] = bspline(mi.Float(ratio[i]), self.knots, self.coeff, self.spline_degree)

        value = color

        return bs, dr.select(active & (bs.pdf > 0), mi.depolarizer(value), mi.Spectrum(0.0))

    def eval(self, ctx, si, wo, active=True):
        if not ctx.is_enabled(mi.BSDFFlags.DiffuseReflection):
            return mi.Spectrum(0.0)
        
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        
        active &= (cos_theta_i > 0) & (cos_theta_o > 0)

        # Bspline implementation
        ratio = (si.wavelengths - 400.0) / (700.0 - 400.0)
        color = mi.unpolarized_spectrum(0.0)
        for i in range(4):
            color[i] = bspline(mi.Float(ratio[i]), self.knots, self.coeff, self.spline_degree)

        value = color * dr.rcp(dr.pi) * cos_theta_o


        return dr.select(active, mi.depolarizer(value), mi.Spectrum(0.0))

    def pdf(self, ctx, si, wo, active=True):
        if not ctx.is_enabled(mi.BSDFFlags.DiffuseReflection):
            return 0.0
        
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        
        pdf_val = mi.warp.square_to_cosine_hemisphere_pdf(wo)
        return dr.select((cos_theta_i > 0) & (cos_theta_o > 0), pdf_val, 0.0)

    def eval_pdf(self, ctx, si, wo, active=True):
        if not ctx.is_enabled(mi.BSDFFlags.DiffuseReflection):
            return mi.Spectrum(0.0), 0.0
        
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        
        active &= (cos_theta_i > 0) & (cos_theta_o > 0)

        # Bspline implementation
        ratio = (si.wavelengths - 400.0) / (700.0 - 400.0)
        color = mi.unpolarized_spectrum(0.0)
        for i in range(4):
            color[i] = bspline(mi.Float(ratio[i]), self.knots, self.coeff, self.spline_degree)

        value = color * dr.rcp(dr.pi) * cos_theta_o

        pdf_val = mi.warp.square_to_cosine_hemisphere_pdf(wo)
        
        return (
            dr.select(active, mi.depolarizer(value), mi.Spectrum(0.0)),
            dr.select(active, pdf_val, 0.0)
        )

    def eval_diffuse_reflectance(self, si, active=True):
        return self.m_reflectance.eval(si, active)

    def to_string(self):
        return f"""smoothBsplineDiffuse[
            reflectance = {self.m_reflectance}
        ]"""

# Register the plugin
mi.register_bsdf("smoothBsplineDiffuse", lambda props: SmoothBsplineDiffuse(props))
from IPython.display import Markdown as md

import numpy as np

def latex_tf(sys, pre="H(s) = "):
    def latex_poly(poly):
        def latex_coef(c, i):
            if np.allclose(c, int(c)):
                c = int(c)
            if i == 0:
                return str(c)
            if c == 1:
                c = ""
            if i == 1:
                return "%s s" % (c,)
            return "%s s^%i" % (c, i)
        return " + ".join(latex_coef(c, len(poly)-i) for i, c in enumerate(poly)
                          if not np.allclose(c, 0))
    return md(r"$%s\frac{%s}{%s}$" % (
        pre, latex_poly(sys.num / sys.gain), latex_poly(sys.den / sys.gain)))
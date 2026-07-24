# -*- coding: utf-8 -*-

<%
import numpy as np
import re
import sys
np.set_printoptions(threshold=sys.maxsize)

%>
## convert an expression from python to julia
<%def name = "py2jl(expr)" filter="trim">
        <%
        expr = expr.replace("**", "^")
        %>
        ${expr}        
</%def>
##         x_j = hcat(v_s[1], v_s[2], )

##     e .= p[1] * (cmin + ( cmax -  cmin) / (1.0 + exp( r * ( midpoint - (x_j[:, 0] - x_j[:, 1])))))

## backfit coupling
<%def name = "backfit_coupling(expr, cvars, n_cterms)" filter="trim">
        <%
        import re

        for idx, cvar in enumerate(cvars):
                expr = re.sub(f"x_j\[.*?{idx}\]", f"v_s[{cvar+1}]", expr)
                expr = re.sub(f"x_i\[.*?{idx}\]", f"v_d[{cvar+1}]", expr)
        %>
        %for i in range(n_cterms):
        e[${i+1}] = p[1] * (${re.sub(f"x_j", f"v_s[{cvars[i]+1}]", re.sub(f"x_i", f"v_d[{cvars[i]+1}]", expr))})
        %endfor

</%def>
## Helper that converts a derived variable of interest into an expression that indexes from trace
<%def name = "gernerate_derived_expresion(var, svars)" filter="trim">
        <%
        for svar in svars:
                 var = var.replace(svar, f"trace[:,[{svars.index(svar)}], :]")
        ##  Use Jax instead of numpy if present
        var = var.replace("np.", "jnp.")
        var = var.replace("numpy.", "jnp.")
        %>
        ${var}
</%def>
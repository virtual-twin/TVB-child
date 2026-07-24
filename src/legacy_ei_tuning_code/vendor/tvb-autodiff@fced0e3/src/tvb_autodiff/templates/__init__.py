# -*- coding: utf-8 -*-


import os

from mako.template import Template
from mako.lookup import TemplateLookup
from mako.exceptions import text_error_template

import autopep8

here = os.path.dirname(os.path.abspath(__file__))


class MakoUtilMix:

    @property
    def lookup(self):
        lookup = TemplateLookup(directories=[here])
        return lookup

    def render_template(self, source, content):
        template = Template(source, lookup=self.lookup, strict_undefined=True)
        try:
            source = template.render(**content)
        except Exception as exc:
            print(text_error_template().render())
            raise exc
        return source

    def insert_line_numbers(self, source):
        lines = source.split('\n')
        numbers = range(1, len(lines) + 1)
        nu_lines = ['%03d\t%s' % (nu, li) for (nu, li) in zip(numbers, lines)]
        nu_source = '\n'.join(nu_lines)
        return nu_source
    
    def build_py_func(self, template_source, content, name='kernel', print_source=True,
            modname=None, fname=None):
        "Build and retrieve one or more Python functions from template."
        source = self.render_template(template_source, content)
        source = autopep8.fix_code(source)
        if print_source:
            print(self.insert_line_numbers(source))
        if fname is not None:
            # check if fname dir exists
            dirname = os.path.dirname(fname)
            if os.path.exists(dirname):
                print("Saving at:")
                print(fname)
                with open(fname, 'w') as fd:
                    fd.write(source)
            else: 
                # warn
                print(f"WARNING: Directory {dirname} does not exist. Not saving simulation")
    
        if modname is not None:
            return self.eval_module(source, name, modname)
        else:
            return self.eval_source(source, name, print_source)

    def eval_source(self, source, name, print_source):
        globals_ = {}
        try:
            exec(source, globals_)
        except Exception as exc:
            if not print_source:
                print(self.insert_line_numbers(source))
            raise exc
        fns = [globals_[n] for n in name.split(',')]
        return fns[0] if len(fns)==1 else fns

    def eval_module(self, source, name, modname):
        here = os.path.abspath(os.path.dirname(__file__))
        genp = os.path.join(here, 'templates', 'generated')
        with open(f'{genp}/{modname}.py', 'w') as fd:
            fd.write(source)
        fullmodname = f'tvb.simulator.backend.templates.generated.{modname}'
        mod = importlib.import_module(fullmodname)
        fns = [getattr(mod,n) for n in name.split(',')]
        return fns[0] if len(fns)==1 else fns

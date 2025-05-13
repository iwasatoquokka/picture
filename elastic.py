# -*- coding: utf-8 -*-
r"""
Nearly incompressible Mooney-Rivlin hyperelastic material model: Cylinder Extension.

Large deformation is described using the total Lagrangian formulation.
This example simulates a cylinder being stretched to over 200% elongation.
"""
from __future__ import print_function
from __future__ import absolute_import
import numpy as nm

from sfepy import data_dir, Mesh, Problem
from sfepy.terms import Term
from sfepy.discrete.conditions import EssentialBC
from sfepy.solvers.ls import Direct
from sfepy.solvers.nls import Newton
from sfepy.postprocess import VTKWriter

# 1. Mesh
# 円柱メッシュを生成します。
radius = 0.5  # 半径
length = 2.0  # 長さ
n_r = 5      # 半径方向の要素数
n_a = 20     # 円周方向の要素数
n_l = 10     # 長さ方向の要素数
mesh = Mesh.from_cylinder(radius=radius, length=length, n_r=n_r, n_a=n_a, n_l=n_l)

filename_mesh = 'cylinder_stretch.vtk' # メッシュの保存パス

# 2. Options
options = {
    'nls': 'newton',
    'ls': 'ls',
    'ts': 'ts',
    'save_times': 'all',
    'post_process_hook': 'calculate_strain_stress',  # カスタムポストプロセス関数を指定
}

# 3. Fields
field_1 = {
    'name': 'displacement',
    'dtype': nm.float64,
    'shape': 3,
    'region': 'Omega',
    'approx_order': 1,
}

# 4. Material
material_1 = {
    'name': 'solid',
    'values': {
        'K': 1e3,  # Bulk modulus
        'mu': 20e0,  # Shear modulus (neo-Hookean)
        'kappa': 10e0,  # Shear modulus (Mooney-Rivlin)
    }
}

# 5. Variables
variables = {
    'u': ('unknown field', 'displacement', 0),
    'v': ('test field', 'displacement', 'u'),
}

# 6. Regions
regions = {
    'Omega': 'all',
    'Bottom': ('vertices in (x < 1e-6)', 'facet'),
    'Top': ('vertices in (x > %f)' % (length - 1e-6), 'facet'),
}

# 7. Boundary Conditions
ebcs = {
    'Bottom': ('Bottom', {'u.all': 0.0}),  # Bottom fixed
    'Top': ('Top', {'u.[1,2]': 0.0}),  # Top: no y, z displacement
}

# 8. Functions (境界条件で利用する関数)
def top_displacement(ts, coor, **kwargs):
    """
    Apply displacement to the top surface to achieve 200% elongation.
    """
    current_length = length + kwargs['displacement']
    new_x = coor[:, 0] + kwargs['displacement']
    return new_x - coor[:, 0]

functions = {
    'top_displacement': (top_displacement,),
}

# 9. カスタムポストプロセス関数
def calculate_strain_stress(out, problem, state, extend=False):
    """
    Calculates and outputs Green-Lagrange strain and Second Piola-Kirchhoff stress.
    """
    from sfepy.base.base import Struct

    strain = problem.evaluate('dw_tl_he_neohook.i.Omega(solid.mu, v, u)',
                              mode='el_avg', term_mode='strain')
    stress = problem.evaluate('dw_tl_he_neohook.i.Omega(solid.mu, v, u)',
                              mode='el_avg', term_mode='stress')

    out['green_lagrange_strain'] = Struct(name='output_data', mode='cell', data=strain, dofs=None)
    out['second_piola_kirchhoff_stress'] = Struct(name='output_data', mode='cell', data=stress, dofs=None)
    return out

# 10. Equations
equations = {
    'equilibrium': """dw_tl_he_neohook.i.Omega( solid.mu, v, u )
                  + dw_tl_he_mooney_rivlin.i.Omega( solid.kappa, v, u )
                  + dw_tl_bulk_penalty.i.Omega( solid.K, v, u ) = 0"""
}

# 11. Solvers
solvers = {
    'ls': ('ls.scipy_direct', {}),
    'nls': ('nls.newton', {
        'i_max': 10,
        'eps_a': 1e-8,
    }),
    'ts': ('ts.simple', {  # Time stepping solver
        't0': 0.0,
        't1': 1.0,
        'n_step': 20,  # Number of time steps
    }),
}

# 12. Problem setup and solution
problem = Problem('cylinder_stretch',
                    equations=equations,
                    variables=variables,
                    domains={'Omega': mesh},  # Use the generated mesh
                    regions=regions,
                    materials={'solid': material_1},
                    ebcs=ebcs,
                    functions=functions,
                    )

# Time stepping loop for displacement control
max_displacement = 2.0 * length  # 200% elongation
n_steps = 20
displacement_values = nm.linspace(0, max_displacement, n_steps + 1)

for i, displacement in enumerate(displacement_values):
    print(f"Step {i + 1}/{n_steps + 1}, Displacement: {displacement:.3f}")

    # Update the essential boundary condition for the top displacement
    bc_top = EssentialBC('Top', regions['Top'], variables['u'].name,
                          lambda ts, coors: top_displacement(ts, coors, displacement=displacement))
    problem.set_bcs([ebcs['Bottom'], bc_top])

    # Solve the problem
    state = problem.solve()

    # キーを確実に存在させる
    strain_stress_data = problem.post_process(state=state, calculate_strain_stress=calculate_strain_stress) # ポストプロセス実行
    if 'green_lagrange_strain' not in strain_stress_data:
        strain_stress_data['green_lagrange_strain'] = None
    if 'second_piola_kirchhoff_stress' not in strain_stress_data:
        strain_stress_data['second_piola_kirchhoff_stress'] = None
    # Save the results (VTK format) - append time step to filename
    filename = 'output_cylinder_stretch_%03d.vtk' % i
    # Create a VTKWriter instance and add data.
    writer = VTKWriter(filename, problem.domain.mesh, append_mode=False)
    writer.write(displacement=state.get_parts('u'),
                   strain=strain_stress_data['green_lagrange_strain'],
                   stress=strain_stress_data['second_piola_kirchhoff_stress'])

print("Simulation completed. Results saved to output_cylinder_stretch_*.vtk")

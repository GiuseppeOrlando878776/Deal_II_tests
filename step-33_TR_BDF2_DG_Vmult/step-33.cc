// @sect3{Include files}

// First a standard set of deal.II includes. Nothing special to comment on
// here:
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/base/mpi.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/solution_transfer.h>

// Then, as mentioned in the introduction, we use various Trilinos packages as
// linear solvers as well as for automatic differentiation. These are in the
// following include files.
//
// Since deal.II provides interfaces to the basic Trilinos matrices,
// preconditioners and solvers, we include them similarly as deal.II linear
// algebra structures.
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>


// Sacado is the automatic differentiation package within Trilinos, which is
// used to find the Jacobian for a fully implicit Newton iteration:
#include <Sacado.hpp>

// And this again is C++:
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <array>

#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/petsc_matrix_free.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/linear_operator.h>

// To end this section, introduce everything in the dealii library into the
// namespace into which the contents of this program will go:
namespace Step33 {
  using namespace dealii;
  namespace LA {
     //using namespace dealii::LinearAlgebraPETSc;
     using namespace dealii::LinearAlgebraTrilinos;
  }

  // @sect3{Euler equation specifics}

  // Here we define the flux function for this particular system of
  // conservation laws, as well as pretty much everything else that's specific
  // to the Euler equations for gas dynamics, for reasons discussed in the
  // introduction. We group all this into a structure that defines everything
  // that has to do with the flux. All members of this structure are static,
  // i.e. the structure has no actual state specified by instance member
  // variables. The better way to do this, rather than a structure with all
  // static members would be to use a namespace -- but namespaces can't be
  // templatized and we want some of the member variables of the structure to
  // depend on the space dimension, which we in our usual way introduce using
  // a template parameter.
  template<int dim>
  struct EulerEquations {
    // @sect4{Component description}

    // First a few variables that describe the various components of our
    // solution vector in a generic way. This includes the number of
    // components in the system (Euler's equations have one entry for momenta
    // in each spatial direction, plus the energy and density components, for
    // a total of <code>dim+2</code> components), as well as functions that
    // describe the index within the solution vector of the first momentum
    // component, the density component, and the energy density
    // component. Note that all these %numbers depend on the space dimension;
    // defining them in a generic way (rather than by implicit convention)
    // makes our code more flexible and makes it easier to later extend it,
    // for example by adding more components to the equations.
    static const unsigned int n_components             = dim + 2;
    static const unsigned int first_momentum_component = 0;
    static const unsigned int density_component        = dim;
    static const unsigned int energy_component         = dim + 1;

    // When generating graphical output way down in this program, we need to
    // specify the names of the solution variables as well as how the various
    // components group into vector and scalar fields. We could describe this
    // there, but in order to keep things that have to do with the Euler
    // equation localized here and the rest of the program as generic as
    // possible, we provide this sort of information in the following two
    // functions:
    static std::vector<std::string> component_names() {
      std::vector<std::string> names(dim, "momentum");
      names.emplace_back("density");
      names.emplace_back("energy_density");

      return names;
    }


    static std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation() {
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
      data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
      data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

      return data_component_interpretation;
    }


    // @sect4{Transformations between variables}

    // Next, we define the gas constant. We will set it to 1.4 in its
    // definition immediately following the declaration of this class (unlike
    // integer variables, like the ones above, static const floating point
    // member variables cannot be initialized within the class declaration in
    // C++). This value of 1.4 is representative of a gas that consists of
    // molecules composed of two atoms, such as air which consists up to small
    // traces almost entirely of $N_2$ and $O_2$.
    static const double gas_gamma;


    // In the following, we will need to compute the kinetic energy, the velocity,
    // the pressure and the spped of sound from a vector of conserved variables. This we can do based on
    // the energy density and the kinetic energy $\frac 12 \rho |\mathbf v|^2
    // = \frac{|\rho \mathbf v|^2}{2\rho}$ (note that the independent
    // variables contain the momentum components $\rho v_i$, not the
    // velocities $v_i$).
    template<typename InputVector>
    static typename InputVector::value_type compute_kinetic_energy(const InputVector& W) {
      typename InputVector::value_type kinetic_energy = 0;
      for(unsigned int d = 0; d < dim; ++d)
        kinetic_energy += W[first_momentum_component + d]*W[first_momentum_component + d];
      kinetic_energy *= 1.0/(2.0*W[density_component]);

      return kinetic_energy;
    }


    template<typename InputVector>
    static typename InputVector::value_type compute_velocity(const InputVector& W) {
      typename InputVector::value_type velocity = 0;
      for(unsigned int d = 0; d < dim; ++d)
        velocity += W[first_momentum_component + d]*W[first_momentum_component + d];
      velocity *= 1.0/(W[density_component]*W[density_component]);

      return std::sqrt(velocity);
    }


    template<typename InputVector>
    static typename InputVector::value_type compute_pressure(const InputVector& W) {
      return (gas_gamma - 1.0)*(W[energy_component] - compute_kinetic_energy(W));
    }


    template<typename InputVector>
    static typename InputVector::value_type compute_speed_of_sound(const InputVector& W) {
      const auto pressure = compute_pressure(W);

      return gas_gamma*pressure/W[density_component];
    }


    // @sect4{EulerEquations::compute_flux_matrix}

    // We define the flux function $F(W)$ as one large matrix.  Each row of
    // this matrix represents a scalar conservation law for the component in
    // that row.  The exact form of this matrix is given in the
    // introduction. Note that we know the size of the matrix: it has as many
    // rows as the system has components, and <code>dim</code> columns; rather
    // than using a FullMatrix object for such a matrix (which has a variable
    // number of rows and columns and must therefore allocate memory on the
    // heap each time such a matrix is created), we use a rectangular array of
    // numbers right away.
    //
    // We templatize the numerical type of the flux function so that we may
    // use the automatic differentiation type here.  Similarly, we will call
    // the function with different input vector data types, so we templatize
    // on it as well:
    template<typename InputVector>
    static void compute_flux_matrix(const InputVector& W,
                                    std::array<std::array<typename InputVector::value_type, dim>,
                                               EulerEquations<dim>::n_components>& flux) {
      // First compute the pressure that appears in the flux matrix, and then
      // compute the first <code>dim</code> columns of the matrix that
      // correspond to the momentum terms:
      const typename InputVector::value_type pressure = compute_pressure(W);

      for(unsigned int d = 0; d < dim; ++d) {
        for(unsigned int e = 0; e < dim; ++e)
          flux[first_momentum_component + d][e] = W[first_momentum_component + d]*
                                                  W[first_momentum_component + e]/W[density_component];

        flux[first_momentum_component + d][d] += pressure;
      }

      // Then the terms for the density (i.e. mass conservation), and, lastly,
      // conservation of energy:
      for(unsigned int d = 0; d < dim; ++d)
        flux[density_component][d] = W[first_momentum_component + d];

      for(unsigned int d = 0; d < dim; ++d)
        flux[energy_component][d] = W[first_momentum_component + d]/W[density_component] *
                                    (W[energy_component] + pressure);
    }


    // @sect4{EulerEquations::compute_normal_flux}

    // On the boundaries of the domain and across hanging nodes we use a
    // numerical flux function to enforce boundary conditions.  This routine
    // is the basic Lax-Friedrich's flux. It's form has also been given already in the introduction:
    template<typename InputVector>
    static void numerical_normal_flux(const Tensor<1, dim>& normal,
                                      const InputVector&    Wplus,
                                      const InputVector&    Wminus,
                                      const double          gamma,
                                      std::array<typename InputVector::value_type, n_components>& normal_flux) {
      std::array<std::array<typename InputVector::value_type, dim>,
                 EulerEquations<dim>::n_components>                 iflux, oflux;

      compute_flux_matrix(Wplus, iflux);
      compute_flux_matrix(Wminus, oflux);

      for(unsigned int di = 0; di < n_components; ++di) {
        normal_flux[di] = 0;
        for(unsigned int d = 0; d < dim; ++d)
          normal_flux[di] += 0.5*(iflux[di][d] + oflux[di][d])*normal[d];

        normal_flux[di] += 0.5*gamma*(Wplus[di] - Wminus[di]);
      }
    }

    // @sect4{EulerEquations::compute_forcing_vector}

    // In the same way as describing the flux function $\mathbf F(\mathbf w)$,
    // we also need to have a way to describe the right hand side forcing
    // term. As mentioned in the introduction, we consider only gravity here,
    // which leads to the specific form $\mathbf G(\mathbf w) = \left(
    // g_1\rho, g_2\rho, g_3\rho, 0, \rho \mathbf g \cdot \mathbf v
    // \right)^T$, shown here for the 3d case. More specifically, we will
    // consider only $\mathbf g=(0,0,-1)^T$ in 3d, or $\mathbf g=(0,-1)^T$ in
    // 2d. This naturally leads to the following function:
    template<typename InputVector>
    static void compute_forcing_vector(const InputVector& W,
                                       std::array<typename InputVector::value_type, n_components>& forcing,
                                       const unsigned int testcase) {
      if(testcase == 0) {
        const double gravity = -1.0;

        for(unsigned int c = 0; c < n_components; ++c) {
          switch(c) {
            case first_momentum_component + dim - 1:
              forcing[c] = gravity*W[density_component];
              break;
            case energy_component:
              forcing[c] = gravity*W[first_momentum_component + dim - 1];
              break;
            default:
              forcing[c] = 0;
          }
        }
      }
      else
        std::fill(forcing.begin(), forcing.end(), 0.0);
    }


    // @sect4{Dealing with boundary conditions}

    // Another thing we have to deal with is boundary conditions. To this end,
    // let us first define the kinds of boundary conditions we currently know
    // how to deal with:
    enum BoundaryKind {
      inflow_boundary,
      outflow_boundary,
      no_penetration_boundary,
      pressure_boundary
    };


    // The next part is to actually decide what to do at each kind of
    // boundary. To this end, remember from the introduction that boundary
    // conditions are specified by choosing a value $\mathbf w^-$ on the
    // outside of a boundary given an inhomogeneity $\mathbf j$ and possibly
    // the solution's value $\mathbf w^+$ on the inside. Both are then passed
    // to the numerical flux $\mathbf H(\mathbf{w}^+, \mathbf{w}^-,
    // \mathbf{n})$ to define boundary contributions to the bilinear form.
    //
    // Boundary conditions can in some cases be specified for each component
    // of the solution vector independently. For example, if component $c$ is
    // marked for inflow, then $w^-_c = j_c$. If it is an outflow, then $w^-_c
    // = w^+_c$. These two simple cases are handled first in the function
    // below.
    //
    // There is a little snag that makes this function unpleasant from a C++
    // language viewpoint: The output vector <code>Wminus</code> will of
    // course be modified, so it shouldn't be a <code>const</code>
    // argument. Yet it is in the implementation below, and needs to be in
    // order to allow the code to compile. The reason is that we call this
    // function at a place where <code>Wminus</code> is of type
    // <code>Table@<2,Sacado::Fad::DFad@<double@> @></code>, this being 2d
    // table with indices representing the quadrature point and the vector
    // component, respectively. We call this function with
    // <code>Wminus[q]</code> as last argument; subscripting a 2d table yields
    // a temporary accessor object representing a 1d vector, just what we want
    // here. The problem is that a temporary accessor object can't be bound to
    // a non-const reference argument of a function, as we would like here,
    // according to the C++ 1998 and 2003 standards (something that will be
    // fixed with the next standard in the form of rvalue references).  We get
    // away with making the output argument here a constant because it is the
    // <i>accessor</i> object that's constant, not the table it points to:
    // that one can still be written to. The hack is unpleasant nevertheless
    // because it restricts the kind of data types that may be used as
    // template argument to this function: a regular vector isn't going to do
    // because that one can not be written to when marked
    // <code>const</code>. With no good solution around at the moment, we'll
    // go with the pragmatic, even if not pretty, solution shown here:
    template<typename DataVector>
    static void compute_Wminus(const std::array<BoundaryKind, n_components>& boundary_kind,
                               const Tensor<1, dim>& normal_vector,
                               const DataVector&     Wplus,
                               const Vector<double>& boundary_values,
                               const DataVector&     Wminus) {
      for(unsigned int c = 0; c < n_components; c++) {
        switch(boundary_kind[c]) {
          case inflow_boundary:
          {
            Wminus[c] = boundary_values(c);
            break;
          }

          case outflow_boundary:
          {
            Wminus[c] = Wplus[c];
            break;
          }


          // Prescribed pressure boundary conditions are a bit more
          // complicated by the fact that even though the pressure is
          // prescribed, we really are setting the energy component here,
          // which will depend on velocity and pressure. So even though this
          // seems like a Dirichlet type boundary condition, we get
          // sensitivities of energy to velocity and density (unless these are
          // also prescribed):
          case pressure_boundary:
          {
            const typename DataVector::value_type density = (boundary_kind[density_component] == inflow_boundary ?
                                                             boundary_values(density_component) :
                                                             Wplus[density_component]);

            typename DataVector::value_type kinetic_energy = 0;
            for(unsigned int d = 0; d < dim; ++d)
              if(boundary_kind[d] == inflow_boundary)
                kinetic_energy += boundary_values(d)*boundary_values(d);
              else
                kinetic_energy += Wplus[d] * Wplus[d];
             kinetic_energy *= 0.5/ density;

             Wminus[c] = boundary_values(c)/(gas_gamma - 1.0) + kinetic_energy;

             break;
          }

          case no_penetration_boundary:
          {
            // We prescribe the velocity (we are dealing with a particular
            // component here so that the average of the velocities is
            // orthogonal to the surface normal.  This creates sensitivities
            // of across the velocity components.
            typename DataVector::value_type vdotn = 0;
            for(unsigned int d = 0; d < dim; d++)
              vdotn += Wplus[d] * normal_vector[d];

            Wminus[c] = Wplus[c] - 2.0 * vdotn * normal_vector[c];
            break;
          }

          default:
            Assert(false, ExcNotImplemented());

        }
      }
    }


    // @sect4{EulerEquations::compute_refinement_indicators}

    // In this class, we also want to specify how to refine the mesh. The
    // class <code>ConservationLaw</code> that will use all the information we
    // provide here in the <code>EulerEquation</code> class is pretty agnostic
    // about the particular conservation law it solves: as doesn't even really
    // care how many components a solution vector has. Consequently, it can't
    // know what a reasonable refinement indicator would be. On the other
    // hand, here we do, or at least we can come up with a reasonable choice:
    // we simply look at the gradient of the density, and compute
    // $\eta_K=\log\left(1+|\nabla\rho(x_K)|\right)$, where $x_K$ is the
    // center of cell $K$.
    //
    // There are certainly a number of equally reasonable refinement
    // indicators, but this one does, and it is easy to compute:
    static void
    compute_refinement_indicators(const DoFHandler<dim>& dof_handler,
                                  const Mapping<dim>&    mapping,
                                  const LA::MPI::Vector& solution,
                                  Vector<double>&        refinement_indicators) {
      const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
      std::vector<unsigned int> dofs(dofs_per_cell);

      const QMidpoint<dim> quadrature_formula;
      const UpdateFlags    update_flags = update_gradients;
      FEValues<dim>        fe_v(mapping, dof_handler.get_fe(),
                                quadrature_formula, update_flags);

      std::vector<std::vector<Tensor<1, dim>>> dU(1, std::vector<Tensor<1, dim>>(n_components));

      for(const auto& cell: dof_handler.active_cell_iterators()) {
        if(cell->is_locally_owned()) {
          const unsigned int cell_no = cell->active_cell_index();
          fe_v.reinit(cell);
          fe_v.get_function_gradients(solution, dU);

          refinement_indicators(cell_no) = std::log(1 + std::sqrt(dU[0][density_component]*dU[0][density_component]));
        }
      }
    }


    // @sect4{EulerEquations::Postprocessor}

    // Finally, we declare a class that implements a postprocessing of data
    // components. The problem this class solves is that the variables in the
    // formulation of the Euler equations we use are in conservative rather
    // than physical form: they are momentum densities $\mathbf m=\rho\mathbf
    // v$, density $\rho$, and energy density $E$. What we would like to also
    // put into our output file are velocities $\mathbf v=\frac{\mathbf
    // m}{\rho}$ and pressure $p=(\gamma-1)(E-\frac{1}{2} \rho |\mathbf
    // v|^2)$.
    //
    // In addition, we would like to add the possibility to generate schlieren
    // plots. Schlieren plots are a way to visualize shocks and other sharp
    // interfaces. The word "schlieren" is a German word that may be
    // translated as "striae" -- it may be simpler to explain it by an
    // example, however: schlieren is what you see when you, for example, pour
    // highly concentrated alcohol, or a transparent saline solution, into
    // water; the two have the same color, but they have different refractive
    // indices and so before they are fully mixed light goes through the
    // mixture along bent rays that lead to brightness variations if you look
    // at it. That's "schlieren". A similar effect happens in compressible
    // flow because the refractive index depends on the pressure (and
    // therefore the density) of the gas.
    //
    // The origin of the word refers to two-dimensional projections of a
    // three-dimensional volume (we see a 2d picture of the 3d fluid). In
    // computational fluid dynamics, we can get an idea of this effect by
    // considering what causes it: density variations. Schlieren plots are
    // therefore produced by plotting $s=|\nabla \rho|^2$; obviously, $s$ is
    // large in shocks and at other highly dynamic places. If so desired by
    // the user (by specifying this in the input file), we would like to
    // generate these schlieren plots in addition to the other derived
    // quantities listed above.
    //
    // The implementation of the algorithms to compute derived quantities from
    // the ones that solve our problem, and to output them into data file,
    // rests on the DataPostprocessor class. It has extensive documentation,
    // and other uses of the class can also be found in step-29. We therefore
    // refrain from extensive comments.
    class Postprocessor : public DataPostprocessor<dim> {
    public:
      Postprocessor(const bool do_schlieren_plot);

      virtual void evaluate_vector_field(const DataPostprocessorInputs::Vector<dim>& inputs,
                                         std::vector<Vector<double>>& computed_quantities) const override;

      virtual std::vector<std::string> get_names() const override;

      virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
      get_data_component_interpretation() const override;

      virtual UpdateFlags get_needed_update_flags() const override;

    private:
      const bool do_schlieren_plot;
    };
  };


  template<int dim>
  const double EulerEquations<dim>::gas_gamma = 1.4;


  template<int dim>
  EulerEquations<dim>::Postprocessor::Postprocessor(const bool do_schlieren_plot):
    do_schlieren_plot(do_schlieren_plot) {}


  // This is the only function worth commenting on. When generating graphical
  // output, the DataOut and related classes will call this function on each
  // cell, with access to values, gradients, Hessians, and normal vectors (in
  // case we're working on faces) at each quadrature point. Note that the data
  // at each quadrature point is itself vector-valued, namely the conserved
  // variables. What we're going to do here is to compute the quantities we're
  // interested in at each quadrature point. Note that for this we can ignore
  // the Hessians ("inputs.solution_hessians") and normal vectors
  // ("inputs.normals").
  template<int dim>
  void EulerEquations<dim>::Postprocessor::evaluate_vector_field(const DataPostprocessorInputs::Vector<dim>& inputs,
                                                                 std::vector<Vector<double>>& computed_quantities) const {
    // At the beginning of the function, let us make sure that all variables
    // have the correct sizes, so that we can access individual vector
    // elements without having to wonder whether we might read or write
    // invalid elements; we also check that the <code>solution_gradients</code>
    // vector only contains data if we really need it (the system knows about
    // this because we say so in the <code>get_needed_update_flags()</code>
    // function below). For the inner vectors, we check that at least the first
    // element of the outer vector has the correct inner size:
    const unsigned int n_quadrature_points = inputs.solution_values.size();

    if(do_schlieren_plot == true)
      Assert(inputs.solution_gradients.size() == n_quadrature_points,
             ExcInternalError());

    Assert(computed_quantities.size() == n_quadrature_points,
           ExcInternalError());

    Assert(inputs.solution_values[0].size() == n_components,
           ExcInternalError());

    if(do_schlieren_plot == true) {
      Assert(computed_quantities[0].size() == dim + 2, ExcInternalError());
    }
    else {
      Assert(computed_quantities[0].size() == dim + 1, ExcInternalError());
    }

    // Then loop over all quadrature points and do our work there. The code
    // should be pretty self-explanatory. The order of output variables is
    // first <code>dim</code> velocities, then the pressure, and if so desired
    // the schlieren plot. Note that we try to be generic about the order of
    // variables in the input vector, using the
    // <code>first_momentum_component</code> and
    // <code>density_component</code> information:
    for(unsigned int q = 0; q < n_quadrature_points; ++q) {
      const double density = inputs.solution_values[q](density_component);

      for(unsigned int d = 0; d < dim; ++d)
        computed_quantities[q](d) = inputs.solution_values[q](first_momentum_component + d)/density;

      computed_quantities[q](dim) = compute_pressure(inputs.solution_values[q]);

      if(do_schlieren_plot == true)
        computed_quantities[q](dim + 1) = inputs.solution_gradients[q][density_component] *
                                          inputs.solution_gradients[q][density_component];
    }
  }


  template<int dim>
  std::vector<std::string> EulerEquations<dim>::Postprocessor::get_names() const {
    std::vector<std::string> names;
    for(unsigned int d = 0; d < dim; ++d)
      names.emplace_back("velocity");
    names.emplace_back("pressure");

    if(do_schlieren_plot == true)
      names.emplace_back("schlieren_plot");

    return names;
  }


  template<int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  EulerEquations<dim>::Postprocessor::get_data_component_interpretation() const {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);

    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    if(do_schlieren_plot == true)
      interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    return interpretation;
  }


  template<int dim>
  UpdateFlags
  EulerEquations<dim>::Postprocessor::get_needed_update_flags() const {
    if(do_schlieren_plot == true)
      return update_values | update_gradients;
    else
      return update_values;
  }


  // @sect3{Run time parameter handling}

  // Our next job is to define a few classes that will contain run-time
  // parameters (for example solver tolerances, number of iterations,
  // stabilization parameter, and the like). One could do this in the main
  // class, but we separate it from that one to make the program more modular
  // and easier to read: Everything that has to do with run-time parameters
  // will be in the following namespace, whereas the program logic is in the
  // main class.
  //
  // We will split the run-time parameters into a few separate structures,
  // which we will all put into a namespace <code>Parameters</code>. Of these
  // classes, there are a few that group the parameters for individual groups,
  // such as for solvers, mesh refinement, or output. Each of these classes
  // have functions <code>declare_parameters()</code> and
  // <code>parse_parameters()</code> that declare parameter subsections and
  // entries in a ParameterHandler object, and retrieve actual parameter
  // values from such an object, respectively. These classes declare all their
  // parameters in subsections of the ParameterHandler.
  //
  // The final class of the following namespace combines all the previous
  // classes by deriving from them and taking care of a few more entries at
  // the top level of the input file, as well as a few odd other entries in
  // subsections that are too short to warrant a structure by themselves.
  //
  // It is worth pointing out one thing here: None of the classes below have a
  // constructor that would initialize the various member variables. This
  // isn't a problem, however, since we will read all variables declared in
  // these classes from the input file (or indirectly: a ParameterHandler
  // object will read it from there, and we will get the values from this
  // object), and they will be initialized this way. In case a certain
  // variable is not specified at all in the input file, this isn't a problem
  // either: The ParameterHandler class will in this case simply take the
  // default value that was specified when declaring an entry in the
  // <code>declare_parameters()</code> functions of the classes below.
  namespace Parameters {
    // @sect4{Parameters::Solver}
    //
    // The first of these classes deals with parameters for the linear inner
    // solver. It offers parameters that indicate which solver to use (GMRES
    // as a solver for general non-symmetric indefinite systems, or a sparse
    // direct solver), the amount of output to be produced, as well as various
    // parameters that tweak the thresholded incomplete LU decomposition
    // (ILUT) that we use as a preconditioner for GMRES.
    //
    // In particular, the ILUT takes the following parameters:
    // - ilut_fill: the number of extra entries to add when forming the ILU
    //   decomposition
    // - ilut_atol, ilut_rtol: When forming the preconditioner, for certain
    //   problems bad conditioning (or just bad luck) can cause the
    //   preconditioner to be very poorly conditioned.  Hence it can help to
    //   add diagonal perturbations to the original matrix and form the
    //   preconditioner for this slightly better matrix.  ATOL is an absolute
    //   perturbation that is added to the diagonal before forming the prec,
    //   and RTOL is a scaling factor $rtol \geq 1$.
    // - ilut_drop: The ILUT will drop any values that have magnitude less
    //   than this value.  This is a way to manage the amount of memory used
    //   by this preconditioner.
    //
    // The meaning of each parameter is also briefly described in the third
    // argument of the ParameterHandler::declare_entry call in
    // <code>declare_parameters()</code>.
    struct Solver {
      enum SolverType {
        gmres,
        direct
      };
      SolverType solver;

      enum OutputType {
        quiet,
        verbose
      };
      OutputType output;

      double linear_residual;
      int    max_iterations;

      double ilut_fill;
      double ilut_atol;
      double ilut_rtol;
      double ilut_drop;

      static void declare_parameters(ParameterHandler& prm);
      void        parse_parameters(ParameterHandler& prm);
    };


    void Solver::declare_parameters(ParameterHandler& prm) {
      prm.enter_subsection("linear solver");
      {
        prm.declare_entry("output", "quiet",
                          Patterns::Selection("quiet|verbose"),
                          "State whether output from solver runs should be printed. "
                          "Choices are <quiet|verbose>.");
        prm.declare_entry("method", "gmres",
                          Patterns::Selection("gmres|direct"),
                          "The kind of solver for the linear system. "
                          "Choices are <gmres|direct>.");
        prm.declare_entry("residual", "1e-10",
                          Patterns::Double(),
                          "Linear solver residual");
        prm.declare_entry("max iters", "300",
                          Patterns::Integer(),
                          "Maximum solver iterations");
        prm.declare_entry("ilut fill", "2",
                          Patterns::Double(),
                          "Ilut preconditioner fill");
        prm.declare_entry("ilut absolute tolerance", "1e-9",
                          Patterns::Double(),
                          "Ilut preconditioner tolerance");
        prm.declare_entry("ilut relative tolerance", "1.1",
                          Patterns::Double(),
                          "Ilut relative tolerance");
        prm.declare_entry("ilut drop tolerance", "1e-10",
                          Patterns::Double(),
                          "Ilut drop tolerance");
      }
      prm.leave_subsection();
    }



    void Solver::parse_parameters(ParameterHandler& prm) {
      prm.enter_subsection("linear solver");
      {
        const std::string op = prm.get("output");
        if(op == "verbose")
          output = verbose;
        else if(op == "quiet")
          output = quiet;
        else
          AssertThrow(false, ExcNotImplemented());

        const std::string sv = prm.get("method");
        if(sv == "direct")
          solver = direct;
        else if (sv == "gmres")
          solver = gmres;
        else
          AssertThrow(false, ExcNotImplemented());

        linear_residual = prm.get_double("residual");
        max_iterations  = prm.get_integer("max iters");
        ilut_fill       = prm.get_double("ilut fill");
        ilut_atol       = prm.get_double("ilut absolute tolerance");
        ilut_rtol       = prm.get_double("ilut relative tolerance");
        ilut_drop       = prm.get_double("ilut drop tolerance");
      }
      prm.leave_subsection();
    }


    // @sect4{Parameters::Refinement}
    //
    // Similarly, here are a few parameters that determine how the mesh is to
    // be refined (and if it is to be refined at all). For what exactly the
    // shock parameters do, see the mesh refinement functions further down.
    struct Refinement {
      bool   do_refine;
      double shock_val;
      double shock_levels;
      int    global_refinements;

      static void declare_parameters(ParameterHandler& prm);
      void        parse_parameters(ParameterHandler& prm);
    };


    void Refinement::declare_parameters(ParameterHandler& prm) {
      prm.enter_subsection("refinement");
      {
        prm.declare_entry("refinement", "true",
                          Patterns::Bool(),
                          "Whether to perform mesh refinement or not");
        prm.declare_entry("global refinements", "5",
                          Patterns::Integer(),
                          "How many global refinements");
        prm.declare_entry("refinement fraction", "0.1",
                          Patterns::Double(),
                          "Fraction of high refinement");
        prm.declare_entry("unrefinement fraction", "0.1",
                          Patterns::Double(),
                          "Fraction of low unrefinement");
        prm.declare_entry("max elements", "1000000",
                          Patterns::Double(),
                          "maximum number of elements");
        prm.declare_entry("shock value", "4.0",
                          Patterns::Double(),
                          "value for shock indicator");
        prm.declare_entry("shock levels", "3.0",
                          Patterns::Double(),
                          "number of shock refinement levels");
      }
      prm.leave_subsection();
    }


    void Refinement::parse_parameters(ParameterHandler& prm) {
      prm.enter_subsection("refinement");
      {
        do_refine          = prm.get_bool("refinement");
        global_refinements = prm.get_integer("global refinements");
        shock_val          = prm.get_double("shock value");
        shock_levels       = prm.get_double("shock levels");
      }
      prm.leave_subsection();
    }


    // @sect4{Parameters::Output}
    //
    // Then a section on output parameters. We offer to produce Schlieren
    // plots (the squared gradient of the density, a tool to visualize shock
    // fronts), and a time interval between graphical output in case we don't
    // want an output file every time step.
    struct Output {
      bool   schlieren_plot;
      double output_step;

      static void declare_parameters(ParameterHandler& prm);
      void        parse_parameters(ParameterHandler& prm);
    };


    void Output::declare_parameters(ParameterHandler& prm) {
      prm.enter_subsection("output");
      {
        prm.declare_entry("schlieren plot", "true",
                          Patterns::Bool(),
                          "Whether or not to produce schlieren plots");
        prm.declare_entry("step", "-1",
                          Patterns::Double(),
                          "Output once per this period");
      }
      prm.leave_subsection();
    }


    void Output::parse_parameters(ParameterHandler& prm) {
      prm.enter_subsection("output");
      {
        schlieren_plot = prm.get_bool("schlieren plot");
        output_step    = prm.get_double("step");
      }
      prm.leave_subsection();
    }


    // @sect4{Parameters::AllParameters}
    //
    // Finally the class that brings it all together. It declares a number of
    // parameters itself, mostly ones at the top level of the parameter file
    // as well as several in section too small to warrant their own
    // classes. It also contains everything that is actually space dimension
    // dependent, like initial or boundary conditions.
    //
    // Since this class is derived from all the ones above, the
    // <code>declare_parameters()</code> and <code>parse_parameters()</code>
    // functions call the respective functions of the base classes as well.
    //
    // Note that this class also handles the declaration of initial and
    // boundary conditions specified in the input file. To this end, in both
    // cases, there are entries like "w_0 value" which represent an expression
    // in terms of $x,y,z$ that describe the initial or boundary condition as
    // a formula that will later be parsed by the FunctionParser
    // class. Similar expressions exist for "w_1", "w_2", etc, denoting the
    // <code>dim+2</code> conserved variables of the Euler system. Similarly,
    // we allow up to <code>max_n_boundaries</code> boundary indicators to be
    // used in the input file, and each of these boundary indicators can be
    // associated with an inflow, outflow, or pressure boundary condition,
    // with homogeneous boundary conditions being specified for each
    // component and each boundary indicator separately.
    //
    // The data structure used to store the boundary indicators is a bit
    // complicated. It is an array of <code>max_n_boundaries</code> elements
    // indicating the range of boundary indicators that will be accepted. For
    // each entry in this array, we store a pair of data in the
    // <code>BoundaryCondition</code> structure: first, an array of size
    // <code>n_components</code> that for each component of the solution
    // vector indicates whether it is an inflow, outflow, or other kind of
    // boundary, and second a FunctionParser object that describes all
    // components of the solution vector for this boundary id at once.
    //
    // The <code>BoundaryCondition</code> structure requires a constructor
    // since we need to tell the function parser object at construction time
    // how many vector components it is to describe. This initialization can
    // therefore not wait till we actually set the formulas the FunctionParser
    // object represents later in
    // <code>AllParameters::parse_parameters()</code>
    //
    // For the same reason of having to tell Function objects their vector
    // size at construction time, we have to have a constructor of the
    // <code>AllParameters</code> class that at least initializes the other
    // FunctionParser object, i.e. the one describing initial conditions.
    template<int dim>
    struct AllParameters : public Solver,
                           public Refinement,
                           public Output {
      static const unsigned int max_n_boundaries = 10;

      struct BoundaryConditions {
        std::array<typename EulerEquations<dim>::BoundaryKind,
                   EulerEquations<dim>::n_components>           kind;

        FunctionParser<dim> values;

        BoundaryConditions();
      };

      //--- Auxiliary class in case exact solution is available
      struct ExactSolution : public Function<dim> {
        ExactSolution(const double time): Function<dim>(dim + 2, time) {}

        virtual double value(const Point<dim>& p,
                             const unsigned int component = 0) const override;

        virtual void   vector_value(const Point<dim>& p,
                                    Vector<double>& values) const override;
      };

      AllParameters(); //--- Default constructor

      AllParameters(ParameterHandler& prm); //--- Constructor

      double time_step, final_time;
      double theta;
      bool   is_stationary;

      std::string time_integration_scheme; //---Extra variable to specify the time integration scheme
      std::string dir;
      std::string mesh_filename;
      unsigned int testcase; //---Extra variable to specify the testcase
      unsigned int fe_degree; //---Extra variable to specify degree of finte element

      FunctionParser<dim> initial_conditions;
      BoundaryConditions  boundary_conditions[max_n_boundaries];
      ExactSolution       exact_solution;

      static void declare_parameters(ParameterHandler& prm);
      void        parse_parameters(ParameterHandler& prm);
    };


    // Constructor of auxiliary struct BoundaryConditions
    template<int dim>
    AllParameters<dim>::BoundaryConditions::BoundaryConditions():
                        values(EulerEquations<dim>::n_components) {
      std::fill(kind.begin(), kind.end(),
                EulerEquations<dim>::no_penetration_boundary);
    }


    // This function will never be called explicitly but it has to be override
    // for the function integrate_difference we will use to compute errors
    template<int dim>
    double AllParameters<dim>::ExactSolution::value(const Point<dim>& x,
                                                    const unsigned int component) const {
      const double t = this->get_time(); //--- This is a member function of the class Function

      Assert(dim == 2, ExcNotImplemented());
      const double beta = 5.0;

      Point<dim> x0;
      x0[0] = 5.0;
      const double gamma_m1    = EulerEquations<dim>::gas_gamma - 1.0;
      const double radius_sqr  = (x - x0).norm_square() - 2.0*(x[0] - x0[0])*t + t*t;
      const double factor      = beta/(2.0*numbers::PI)*std::exp(1.0 - radius_sqr);
      const double density_log = std::log2(std::abs(1.0 - gamma_m1/EulerEquations<dim>::gas_gamma*0.25*factor*factor));
      const double density     = std::exp2(density_log/gamma_m1);
      const double u           = 1.0 - factor*(x[1] - x0[1]);
      const double v           = factor*(x[0] - t - x0[0]);

      if(component == EulerEquations<dim>::density_component)
        return density;
      else if(component == EulerEquations<dim>::first_momentum_component)
        return density*u;
      else if(component == EulerEquations<dim>::first_momentum_component + 1)
        return density*v;
      else {
        const double pressure = std::exp2(density_log*(EulerEquations<dim>::gas_gamma/gamma_m1));
        return pressure/gamma_m1 + 0.5*(density*u*u + density*v*v);
      }
    }


    // This function will never be called explicitly but it has to be override
    // for the function integrate_difference we will use to compute errors
    template<int dim>
    void AllParameters<dim>::ExactSolution::vector_value(const Point<dim>& x,
                                                         Vector<double>& values) const {
      const double t = this->get_time(); //--- This is a function of the class Function

      Assert(dim == 2, ExcNotImplemented());
      AssertThrow(values.size() == EulerEquations<dim>::n_components, ExcMessage("Wrong number of components"));

      const double beta = 5.0;

      Point<dim> x0;
      x0[0] = 5.0;
      const double gamma_m1    = EulerEquations<dim>::gas_gamma - 1.0;
      const double radius_sqr  = (x - x0).norm_square() - 2.0*(x[0] - x0[0])*t + t*t;
      const double factor      = beta/(2.0*numbers::PI)*std::exp(1.0 - radius_sqr);
      const double density_log = std::log2(std::abs(1.0 - gamma_m1/EulerEquations<dim>::gas_gamma*0.25*factor*factor));
      const double density     = std::exp2(density_log/gamma_m1);
      const double u           = 1.0 - factor*(x[1] - x0[1]);
      const double v           = factor*(x[0] - t - x0[0]);
      const double pressure    = std::exp2(density_log*(EulerEquations<dim>::gas_gamma/gamma_m1));
      const double E           = pressure/gamma_m1 + 0.5*(density*u*u + density*v*v);

      values[EulerEquations<dim>::density_component] = density;
      values[EulerEquations<dim>::first_momentum_component] = density*u;
      values[EulerEquations<dim>::first_momentum_component + 1] = density*v;
      values[EulerEquations<dim>::energy_component] = E;
    }


    // Default constructor of struct AllParameters
    template<int dim>
    AllParameters<dim>::AllParameters(): time_step(1.),
                                         final_time(1.),
                                         theta(.5),
                                         is_stationary(true),
                                         testcase(0),
                                         fe_degree(1),
                                         initial_conditions(EulerEquations<dim>::n_components),
                                         exact_solution(ExactSolution(0.0)) {}

    // Struct AllParameters constructor
    template<int dim>
    AllParameters<dim>::AllParameters(ParameterHandler& prm): time_step(1.),
                                                              final_time(1.),
                                                              theta(.5),
                                                              is_stationary(true),
                                                              testcase(0),
                                                              fe_degree(1),
                                                              initial_conditions(EulerEquations<dim>::n_components),
                                                              exact_solution(ExactSolution(0.0)) {
      parse_parameters(prm);
    }


    // Function to declare parameters to be read
    template<int dim>
    void AllParameters<dim>::declare_parameters(ParameterHandler& prm) {
      prm.declare_entry("testcase", "0",
                        Patterns::Integer(0,1),
                        "specification of testcase between "
                        "step-33 (default) and step-67");

      prm.declare_entry("mesh", "grid.inp",
                         Patterns::Anything(),
                        "intput file name");

      prm.declare_entry("degree", "1",
                         Patterns::Integer(0,5),
                         "Finite element degree");

      prm.enter_subsection("time stepping");
      {
        prm.declare_entry("time step", "0.1",
                          Patterns::Double(0),
                          "simulation time step");
        prm.declare_entry("final time", "10.0",
                          Patterns::Double(0),
                          "simulation end time");
        //--- Add a string to specify time integration
        prm.declare_entry("time integration scheme",
                          "Theta_Method",
                          Patterns::Selection("Theta_Method|TR_BDF2"),
                          "specification of time integration scheme between "
                          "TR_BDF2 and Theta method (default)");
        prm.declare_entry("saving directory", "SimTest");

        prm.declare_entry("theta scheme value", "0.5",
                          Patterns::Double(0, 1),
                          "value for theta that interpolated between explicit "
                          "Euler (theta=0), Crank-Nicolson (theta=0.5), and "
                          "implicit Euler (theta=1).");
      }
      prm.leave_subsection();


      for(unsigned int b = 0; b < max_n_boundaries; ++b) {
        prm.enter_subsection("boundary_" + Utilities::int_to_string(b));
        {
          prm.declare_entry("no penetration", "false",
                            Patterns::Bool(),
                            "whether the named boundary allows gas to "
                            "penetrate or is a rigid wall");

          for(unsigned int di = 0; di < EulerEquations<dim>::n_components; ++di) {
            prm.declare_entry("w_" + Utilities::int_to_string(di), "outflow",
                              Patterns::Selection("inflow|outflow|pressure"),
                              "<inflow|outflow|pressure>");

            prm.declare_entry("w_" + Utilities::int_to_string(di) + " value", "0.0",
                              Patterns::Anything(),
                              "expression in x,y,z");
          }
        }
        prm.leave_subsection();
      }

      prm.enter_subsection("initial condition");
      {
        for(unsigned int di = 0; di < EulerEquations<dim>::n_components; ++di)
          prm.declare_entry("w_" + Utilities::int_to_string(di) + " value",
                            "0.0", Patterns::Anything(),
                            "expression in x,y,z");
      }
      prm.leave_subsection();

      Parameters::Solver::declare_parameters(prm);
      Parameters::Refinement::declare_parameters(prm);
      Parameters::Output::declare_parameters(prm);
    }


    // Auxiliary function to parse declared parameters
    template<int dim>
    void AllParameters<dim>::parse_parameters(ParameterHandler& prm) {
      testcase = prm.get_integer("testcase");
      fe_degree = prm.get_integer("degree");
      if(testcase == 0)
        mesh_filename = prm.get("mesh");

      prm.enter_subsection("time stepping");
      {
        time_step = prm.get_double("time step");
        if(time_step == 0.0) {
          is_stationary = true;
          time_step     = 1.0;
          final_time    = 1.0;
        }
        else
          is_stationary = false;

        final_time = prm.get_double("final time");
        time_integration_scheme = prm.get("time integration scheme"); //--- Read the time integration scheme chosen
                                                                      //--- and check the correctness
        dir = prm.get("saving directory");
        theta = prm.get_double("theta scheme value");
      }
      prm.leave_subsection();

      if(testcase == 0) {
        for(unsigned int boundary_id = 0; boundary_id < max_n_boundaries; ++boundary_id) {
          prm.enter_subsection("boundary_" + Utilities::int_to_string(boundary_id));
          {
            std::vector<std::string> expressions(EulerEquations<dim>::n_components, "0.0");

            const bool no_penetration = prm.get_bool("no penetration");

            for(unsigned int di = 0; di < EulerEquations<dim>::n_components; ++di) {
              const std::string boundary_type = prm.get("w_" + Utilities::int_to_string(di));

              if((di < dim) && (no_penetration == true))
                boundary_conditions[boundary_id].kind[di] = EulerEquations<dim>::no_penetration_boundary;
              else if(boundary_type == "inflow")
                boundary_conditions[boundary_id].kind[di] = EulerEquations<dim>::inflow_boundary;
              else if(boundary_type == "pressure")
                boundary_conditions[boundary_id].kind[di] = EulerEquations<dim>::pressure_boundary;
              else if(boundary_type == "outflow")
                boundary_conditions[boundary_id].kind[di] = EulerEquations<dim>::outflow_boundary;
              else
                AssertThrow(false, ExcNotImplemented());

              expressions[di] = prm.get("w_" + Utilities::int_to_string(di) + " value");
            }

            boundary_conditions[boundary_id].values.initialize(FunctionParser<dim>::default_variable_names(),
                                                              expressions, std::map<std::string, double>());
          }
          prm.leave_subsection();
        }

        prm.enter_subsection("initial condition");
        {
          std::vector<std::string> expressions(EulerEquations<dim>::n_components, "0.0");
          for(unsigned int di = 0; di < EulerEquations<dim>::n_components; di++)
            expressions[di] = prm.get("w_" + Utilities::int_to_string(di) + " value");
          initial_conditions.initialize(FunctionParser<dim>::default_variable_names(),
                                        expressions, std::map<std::string, double>());
        }
        prm.leave_subsection();
      }
      else
        std::fill(boundary_conditions[0].kind.begin(), boundary_conditions[0].kind.end(),
                  EulerEquations<dim>::inflow_boundary);

      Parameters::Solver::parse_parameters(prm);
      Parameters::Refinement::parse_parameters(prm);
      if(testcase == 1 && do_refine == true)
        AssertThrow(false, ExcNotImplemented());
      Parameters::Output::parse_parameters(prm);
    }
  } // namespace Parameters


  // @sect3{Conservation law operator class}

  // Here we create an operator that will implement the matrix-vector effect.
  // In this way we shuold be able to avoid the storing of the Jacobian matrix (not so
  // sofisticated like matrix-free but a good alternative, especially in approaches
  // that employ the automatic differentiation)
  template<int dim>
  class ConservationLawOperator {
  public:
    ConservationLawOperator(const DoFHandler<dim>& dof_handler,
                            const MappingQ1<dim>&  mapping,
                            const QGauss<dim>&     quadrature,
                            const QGauss<dim - 1>& face_quadrature,
                            TimerOutput&           time_table,
                            ParameterHandler&      prm,
                            LA::MPI::Vector&       locally_relevant_solution);

    void set_TR_BDF2_stage(const unsigned int stage);

    void set_current_time(const double current_time);

    void set_lambda_old(const double lambda_old);

    void assemble_cell_term(const FEValues<dim>&                              fe_v,
                            const Vector<double>&                             local_current_solution,
                            const std::vector<types::global_dof_index>&       dof_indices,
                            LinearAlgebra::distributed::Vector<double>&       dst,
                            const LinearAlgebra::distributed::Vector<double>& src) const;
    void assemble_face_term(const unsigned int                                face_no,
                            const FEFaceValuesBase<dim>&                      fe_v,
                            const FEFaceValuesBase<dim>&                      fe_v_neighbor,
                            const bool                                        external_face,
                            const unsigned int                                boundary_id,
                            const Vector<double>&                             local_current_solution,
                            const Vector<double>&                             local_current_solution_neighbor,
                            const std::vector<types::global_dof_index>&       dof_indices,
                            const std::vector<types::global_dof_index>&       dof_indices_neighbor,
                            LinearAlgebra::distributed::Vector<double>&       dst,
                            const LinearAlgebra::distributed::Vector<double>& src) const;

    void vmult(LinearAlgebra::distributed::Vector<double>&       dst,
               const LinearAlgebra::distributed::Vector<double>& src) const;

  private:
    const DoFHandler<dim>& dof_handler;
    const MappingQ1<dim>&  mapping;
    const QGauss<dim>&     quadrature;
    const QGauss<dim - 1>& face_quadrature;
    TimerOutput&           time_table;
    Parameters::AllParameters<dim> parameters;  //--- Auxiliary variable to read parameters

    double             lambda_old; //--- Extra parameter to save old Lax-Friedrichs stability (for explicit caching)
    double             gamma_2; //--- Extra parameter of TR_BDF2
    double             gamma_3; //--- Extra parameter of TR_BDF2
    unsigned int       TR_BDF2_stage; //--- Extra parameter for counting stage TR_BDF2

    LA::MPI::Vector&   locally_relevant_solution;
  };

  // Class constructor
  template<int dim>
  ConservationLawOperator<dim>::ConservationLawOperator(const DoFHandler<dim>& dof_handler,
                                                        const MappingQ1<dim>&  mapping,
                                                        const QGauss<dim>&     quadrature,
                                                        const QGauss<dim - 1>& face_quadrature,
                                                        TimerOutput&           time_table,
                                                        ParameterHandler&      prm,
                                                        LA::MPI::Vector&       locally_relevant_solution):
    dof_handler(dof_handler), mapping(mapping), quadrature(quadrature), face_quadrature(face_quadrature), time_table(time_table),
    parameters(prm), locally_relevant_solution(locally_relevant_solution) {
      //--- Compute the other two parameters of TR_BDF2 scheme
      if(parameters.time_integration_scheme == "TR_BDF2") {
        parameters.theta = 1.0 - std::sqrt(2)/2.0;
        gamma_2 = (1.0 - 2.0*parameters.theta)/(2.0*(1.0 - parameters.theta));
        gamma_3 = (1.0 - gamma_2)/(2.0*parameters.theta);
        TR_BDF2_stage = 1;
      }
  }

  // Set stage for TR_BDF2 (fundamental because the effectvve value is known only at
  // runtime and so has to be called by ConservationLaw class)
  template<int dim>
  void ConservationLawOperator<dim>::set_TR_BDF2_stage(const unsigned int stage) {
    Assert(stage == 1 || stage == 2, ExcInternalError());
    TR_BDF2_stage = stage;
  }

  // Set time for boundary conditions (fundamental because the effectvve value is known only at
  // runtime and so has to be called by ConservationLaw class)
  template<int dim>
  void ConservationLawOperator<dim>::set_current_time(const double current_time) {
    if(parameters.testcase == 1)
      parameters.exact_solution.set_time(current_time);
  }

  // Set lambda_old for LF (fundamental because the effectvve value is known only at
  // runtime and so has to be called by ConservationLaw class)
  template<int dim>
  void ConservationLawOperator<dim>::set_lambda_old(const double lambda_old) {
    this->lambda_old = lambda_old;
  }

  // Assemble cell term
  template<int dim>
  void ConservationLawOperator<dim>::assemble_cell_term(const FEValues<dim>&                              fe_v,
                                                        const Vector<double>&                             local_current_solution,
                                                        const std::vector<types::global_dof_index>&       dof_indices,
                                                        LinearAlgebra::distributed::Vector<double>&       dst,
                                                        const LinearAlgebra::distributed::Vector<double>& src) const {
    TimerOutput::Scope t(time_table, "Assemble cell term");

    const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
    const unsigned int n_q_points    = fe_v.n_quadrature_points;

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_src(dofs_per_cell), cell_dst(dofs_per_cell);
    cell_matrix = 0;

    AffineConstraints<double> constraints;

    Table<2, Sacado::Fad::DFad<double>> W(n_q_points, EulerEquations<dim>::n_components);

    std::vector<Sacado::Fad::DFad<double>> independent_local_dof_values(dofs_per_cell);
    for(unsigned int i = 0; i < dofs_per_cell; ++i) {
      independent_local_dof_values[i] = local_current_solution[i];
      independent_local_dof_values[i].diff(i, dofs_per_cell);
    }

    for(unsigned int q = 0; q < n_q_points; ++q)
      for(unsigned int c = 0; c < EulerEquations<dim>::n_components; ++c)
        W[q][c] = 0;

    for(unsigned int q = 0; q < n_q_points; ++q) {
      for(unsigned int i = 0; i < dofs_per_cell; ++i) {
        const unsigned int c = fe_v.get_fe().system_to_component_index(i).first;
        W[q][c] += independent_local_dof_values[i]*fe_v.shape_value_component(i, q, c);
      }
    }

    std::vector<std::array<std::array<Sacado::Fad::DFad<double>, dim>,
                           EulerEquations<dim>::n_components>>          flux(n_q_points);

    std::vector<std::array<Sacado::Fad::DFad<double>, EulerEquations<dim>::n_components>> forcing(n_q_points);

    for(unsigned int q = 0; q < n_q_points; ++q) {
      EulerEquations<dim>::compute_flux_matrix(W[q], flux[q]);
      EulerEquations<dim>::compute_forcing_vector(W[q], forcing[q], parameters.testcase);
    }

    for(unsigned int i = 0; i < dofs_per_cell; ++i) {
      Sacado::Fad::DFad<double> R_i = 0;

      const unsigned int component_i = fe_v.get_fe().system_to_component_index(i).first;

      for(unsigned int point = 0; point < fe_v.n_quadrature_points; ++point) {
        //--- Reorganize residual computation
        if(parameters.time_integration_scheme == "Theta_Method") {
          if(parameters.is_stationary == false)
            R_i += 1.0/parameters.time_step *
                   W[point][component_i] *
                   fe_v.shape_value_component(i, point, component_i) *
                   fe_v.JxW(point);

          for(unsigned int d = 0; d < dim; d++)
            R_i -= parameters.theta*flux[point][component_i][d]*
                   fe_v.shape_grad_component(i, point, component_i)[d] *
                   fe_v.JxW(point);

          R_i -= parameters.theta*forcing[point][component_i]*
                 fe_v.shape_value_component(i, point, component_i) *
                 fe_v.JxW(point);
        }
        //--- Stages of TR-BDF2
        else {
          //--- First stage of TR_BDF2
          if(TR_BDF2_stage == 1) {
            if(parameters.is_stationary == false)
              R_i += 1.0/parameters.time_step *
                     W[point][component_i] *
                     fe_v.shape_value_component(i, point, component_i) *
                     fe_v.JxW(point);

            for(unsigned int d = 0; d < dim; d++)
              R_i -= parameters.theta*flux[point][component_i][d] *
                     fe_v.shape_grad_component(i, point, component_i)[d] *
                     fe_v.JxW(point);

            R_i -= parameters.theta*forcing[point][component_i] *
                   fe_v.shape_value_component(i, point, component_i) *
                   fe_v.JxW(point);
          }
          //--- Second stage of TR_BDF2
          else if(TR_BDF2_stage == 2) {
            if(parameters.is_stationary == false)
              R_i += 1.0/parameters.time_step *
                     W[point][component_i] *
                     fe_v.shape_value_component(i, point, component_i) *
                     fe_v.JxW(point);

            for(unsigned int d = 0; d < dim; d++)
              R_i -= gamma_2*flux[point][component_i][d] *
                     fe_v.shape_grad_component(i, point, component_i)[d] *
                     fe_v.JxW(point);

            R_i -=  gamma_2*forcing[point][component_i] *
                    fe_v.shape_value_component(i, point, component_i) *
                    fe_v.JxW(point);
          }
        }
      }

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        cell_matrix(i, j) += R_i.fastAccessDx(j);
    }
    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      cell_src(i) = src(dof_indices[i]);
    cell_matrix.vmult(cell_dst, cell_src);
    constraints.distribute_local_to_global(cell_dst, dof_indices, dst);
    dst.compress(VectorOperation::add);
  }

  // Assemble face term
  template<int dim>
  void ConservationLawOperator<dim>::assemble_face_term(const unsigned int           face_no,
                                                        const FEFaceValuesBase<dim>& fe_v,
                                                        const FEFaceValuesBase<dim>& fe_v_neighbor,
                                                        const bool                   external_face,
                                                        const unsigned int           boundary_id,
                                                        const Vector<double>&        local_current_solution,
                                                        const Vector<double>&        local_current_solution_neighbor,
                                                        const std::vector<types::global_dof_index>&       dof_indices,
                                                        const std::vector<types::global_dof_index>&       dof_indices_neighbor,
                                                        LinearAlgebra::distributed::Vector<double>&       dst,
                                                        const LinearAlgebra::distributed::Vector<double>& src) const {
    TimerOutput::Scope t(time_table, "Assemble face term");

    const unsigned int n_q_points    = fe_v.n_quadrature_points;
    const unsigned int dofs_per_cell = fe_v.dofs_per_cell;

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell),
                       aux_cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_src(dofs_per_cell), cell_dst(dofs_per_cell);
    cell_matrix = 0;
    aux_cell_matrix = 0;

    AffineConstraints<double> constraints;

    std::vector<Sacado::Fad::DFad<double>> independent_local_dof_values(dofs_per_cell),
                                           independent_neighbor_dof_values(external_face == false ?
                                                                           dofs_per_cell : 0);

    const unsigned int n_independent_variables = (external_face == false ? dofs_per_cell + dofs_per_cell : dofs_per_cell);

    for(unsigned int i = 0; i < dofs_per_cell; i++) {
      independent_local_dof_values[i] = local_current_solution[i];
      independent_local_dof_values[i].diff(i, n_independent_variables);
    }

    if(external_face == false) {
      for(unsigned int i = 0; i < dofs_per_cell; i++) {
        independent_neighbor_dof_values[i] = local_current_solution_neighbor[i];
        independent_neighbor_dof_values[i].diff(i + dofs_per_cell, n_independent_variables);
      }
    }

    Table<2, Sacado::Fad::DFad<double>> Wplus(n_q_points, EulerEquations<dim>::n_components),
                                        Wminus(n_q_points, EulerEquations<dim>::n_components);

    for(unsigned int q = 0; q < n_q_points; ++q) {
      for(unsigned int i = 0; i < dofs_per_cell; ++i) {
        const unsigned int component_i = fe_v.get_fe().system_to_component_index(i).first;
        Wplus[q][component_i] += independent_local_dof_values[i] *
                                 fe_v.shape_value_component(i, q, component_i);
      }
    }

    // Computing "opposite side" is a bit more complicated. If this is
    // an internal face, we can compute it as above by simply using the
    // independent variables from the neighbor:
    if(external_face == false) {
      for(unsigned int q = 0; q < n_q_points; ++q) {
        for(unsigned int i = 0; i < dofs_per_cell; ++i) {
          const unsigned int component_i = fe_v_neighbor.get_fe().system_to_component_index(i).first;
          Wminus[q][component_i] += independent_neighbor_dof_values[i] *
                                    fe_v_neighbor.shape_value_component(i, q, component_i);
        }
      }
    }
    else {
      Assert(boundary_id < Parameters::AllParameters<dim>::max_n_boundaries,
             ExcIndexRange(boundary_id, 0, Parameters::AllParameters<dim>::max_n_boundaries));

      std::vector<Vector<double>> boundary_values(n_q_points, Vector<double>(EulerEquations<dim>::n_components));
      if(parameters.testcase == 0)
        parameters.boundary_conditions[boundary_id].values.vector_value_list(fe_v.get_quadrature_points(), boundary_values);
      else
        parameters.exact_solution.vector_value_list(fe_v.get_quadrature_points(), boundary_values);

      for(unsigned int q = 0; q < n_q_points; q++) {
        // Here we assume that boundary type, boundary normal vector and
        // boundary data values maintain the same during time advancing.
        EulerEquations<dim>::compute_Wminus(parameters.boundary_conditions[boundary_id].kind,
                                            fe_v.normal_vector(q), Wplus[q], boundary_values[q], Wminus[q]);
      }
    }

    std::vector<std::array<Sacado::Fad::DFad<double>, EulerEquations<dim>::n_components>> normal_fluxes(n_q_points);

    for(unsigned int q = 0; q < n_q_points; ++q)
      EulerEquations<dim>::numerical_normal_flux(fe_v.normal_vector(q), Wplus[q], Wminus[q], lambda_old, normal_fluxes[q]);

    for(unsigned int i = 0; i < dofs_per_cell; ++i) {
      if(fe_v.get_fe().has_support_on_face(i, face_no) == true) {
        Sacado::Fad::DFad<double> R_i = 0;

        for(unsigned int point = 0; point < n_q_points; ++point) {
          const unsigned int component_i = fe_v.get_fe().system_to_component_index(i).first;

          //--- Reorganizing contributions
          if(parameters.time_integration_scheme == "Theta_Method") {
            R_i += parameters.theta*normal_fluxes[point][component_i]*
                   fe_v.shape_value_component(i, point, component_i) *
                   fe_v.JxW(point);
          }
          //--- TR_BDF2 assembling
          else {
            //--- First stage of TR_BDF2
            if(TR_BDF2_stage == 1) {
              R_i += parameters.theta*normal_fluxes[point][component_i] *
                     fe_v.shape_value_component(i, point, component_i) *
                     fe_v.JxW(point);
            }
            //--- Second stage of TR_BDF2
            else if(TR_BDF2_stage == 2) {
              R_i += gamma_2*normal_fluxes[point][component_i]*
                     fe_v.shape_value_component(i, point, component_i) *
                     fe_v.JxW(point);
            }
          }
        }

        for(unsigned int j = 0; j < dofs_per_cell; ++j)
          cell_matrix(i,j) += R_i.fastAccessDx(j);

        if(external_face == false) {
          for(unsigned int j = 0; j < dofs_per_cell; ++j)
            aux_cell_matrix(i, j) += R_i.fastAccessDx(dofs_per_cell + j);
        }
      }
    }
    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      cell_src(i) = src(dof_indices[i]);
    cell_matrix.vmult(cell_dst, cell_src);
    constraints.distribute_local_to_global(cell_dst, dof_indices, dst);
    dst.compress(VectorOperation::add);
    if(external_face == false) {
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        cell_src(i) = src(dof_indices_neighbor[i]);
      aux_cell_matrix.vmult(cell_dst, cell_src);
      constraints.distribute_local_to_global(cell_dst, dof_indices_neighbor, dst);
      dst.compress(VectorOperation::add);
    }
  }

  // Now we compute the action of the operator, namely the matrix-vector operator
  template<int dim>
  void ConservationLawOperator<dim>::vmult(LinearAlgebra::distributed::Vector<double>& dst,
                                           const LinearAlgebra::distributed::Vector<double>& src) const {
    TimerOutput::Scope t(time_table, "Action of linear operator");

    const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices_neighbor(dofs_per_cell);

    dst = 0;

    const UpdateFlags update_flags = update_values | update_gradients |
                                     update_quadrature_points | update_JxW_values,
                      face_update_flags = update_values | update_quadrature_points |
                                          update_JxW_values |
                                          update_normal_vectors,
                      neighbor_face_update_flags = update_values;

    FEValues<dim>        fe_v(mapping, dof_handler.get_fe(), quadrature, update_flags);
    FEFaceValues<dim>    fe_v_face(mapping, dof_handler.get_fe(), face_quadrature, face_update_flags);
    FESubfaceValues<dim> fe_v_subface(mapping, dof_handler.get_fe(), face_quadrature, face_update_flags);
    FEFaceValues<dim>    fe_v_face_neighbor(mapping, dof_handler.get_fe(), face_quadrature, neighbor_face_update_flags);
    FESubfaceValues<dim> fe_v_subface_neighbor(mapping, dof_handler.get_fe(), face_quadrature, neighbor_face_update_flags);

    Vector<double> local_current_solution(dofs_per_cell),
                   local_current_solution_neighbor(dofs_per_cell);

    // Then loop over all cells, initialize the FEValues object for the
    // current cell and call the function that assembles the problem on this
    // cell.
    src.update_ghost_values();
    for(const auto& cell: dof_handler.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        fe_v.reinit(cell);
        cell->get_dof_values(locally_relevant_solution, local_current_solution);
        cell->get_dof_indices(dof_indices);

        assemble_cell_term(fe_v, local_current_solution, dof_indices, dst, src);

        // Then loop over all the faces of this cell.  If a face is part of
        // the external boundary, then assemble boundary conditions there:
        for(unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no) {
          if(cell->at_boundary(face_no)) {
            fe_v_face.reinit(cell, face_no);
            assemble_face_term(face_no, fe_v_face, fe_v_face,
                               true, cell->face(face_no)->boundary_id(),
                               local_current_solution, local_current_solution,
                               dof_indices, dof_indices, dst, src);
          }

          // The alternative is that we are dealing with an internal face.
          else {
            if(cell->neighbor(face_no)->has_children()) {
              const unsigned int neighbor2 = cell->neighbor_of_neighbor(face_no);

              for(unsigned int subface_no = 0; subface_no < cell->face(face_no)->n_children(); ++subface_no) {
                const typename DoFHandler<dim>::active_cell_iterator
                neighbor_child = cell->neighbor_child_on_subface(face_no, subface_no);

                Assert(neighbor_child->face(neighbor2) == cell->face(face_no)->child(subface_no), ExcInternalError());
                Assert(neighbor_child->has_children() == false, ExcInternalError());

                fe_v_subface.reinit(cell, face_no, subface_no);
                fe_v_face_neighbor.reinit(neighbor_child, neighbor2);
                neighbor_child->get_dof_values(locally_relevant_solution, local_current_solution_neighbor);
                neighbor_child->get_dof_indices(dof_indices_neighbor);

                assemble_face_term(face_no, fe_v_subface, fe_v_face_neighbor,
                                   false, numbers::invalid_unsigned_int,
                                   local_current_solution, local_current_solution_neighbor,
                                   dof_indices, dof_indices_neighbor, dst, src);
              }
            }

            // The other possibility we have to care for is if the neighbor
            // is coarser than the current cell:
            else if(cell->neighbor(face_no)->level() != cell->level()) {
              const typename DoFHandler<dim>::cell_iterator neighbor = cell->neighbor(face_no);
              Assert(neighbor->level() == cell->level() - 1, ExcInternalError());

              const std::pair<unsigned int, unsigned int> faceno_subfaceno = cell->neighbor_of_coarser_neighbor(face_no);
              const unsigned int neighbor_face_no = faceno_subfaceno.first,
                                 neighbor_subface_no = faceno_subfaceno.second;

              Assert(neighbor->neighbor_child_on_subface(neighbor_face_no, neighbor_subface_no) == cell,
                     ExcInternalError());

              fe_v_face.reinit(cell, face_no);
              fe_v_subface_neighbor.reinit(neighbor, neighbor_face_no, neighbor_subface_no);
              neighbor->get_dof_values(locally_relevant_solution, local_current_solution_neighbor);
              neighbor->get_dof_indices(dof_indices_neighbor);

              assemble_face_term(face_no, fe_v_face, fe_v_subface_neighbor,
                                 false, numbers::invalid_unsigned_int,
                                 local_current_solution, local_current_solution_neighbor,
                                 dof_indices, dof_indices_neighbor, dst, src);
            }
            // Same refinement level
            else {
              fe_v_face.reinit(cell, face_no);
              const typename DoFHandler<dim>::cell_iterator neighbor = cell->neighbor(face_no);
              const unsigned int other_face_no = cell->neighbor_of_neighbor(face_no);
              fe_v_face_neighbor.reinit(neighbor, other_face_no);
              neighbor->get_dof_values(locally_relevant_solution, local_current_solution_neighbor);
              neighbor->get_dof_indices(dof_indices_neighbor);

              assemble_face_term(face_no, fe_v_face, fe_v_face_neighbor,
                                 false, numbers::invalid_unsigned_int,
                                 local_current_solution, local_current_solution_neighbor,
                                 dof_indices, dof_indices_neighbor, dst, src);
            }
          }
        }
      }
    }
  }


  // @sect3{Conservation law operator class}

  // Here we create an operator that will implement the matrix-vector effect.
  // In this way we shuold be able to avoid the storing of the Jacobian matrix (not so
  // sofisticated like matrix-free but a good alternative, especially in approaches
  // that employ the automatic differentiation)
  template<int dim>
  class ConservationLawPreconditioner {
  public:
    ConservationLawPreconditioner(const DoFHandler<dim>& dof_handler,
                                  const MappingQ1<dim>&  mapping,
                                  const QGauss<dim>&     quadrature,
                                  const QGauss<dim - 1>& face_quadrature,
                                  TimerOutput&           time_table,
                                  ParameterHandler&      prm,
                                  LA::MPI::Vector&       locally_relevant_solution);

    void set_TR_BDF2_stage(const unsigned int stage);

    void set_current_time(const double current_time);

    void set_lambda_old(const double lambda_old);

    void assemble_cell_term(const FEValues<dim>&                              fe_v,
                            const Vector<double>&                             local_current_solution,
                            const std::vector<types::global_dof_index>&       dof_indices) const;
    void assemble_face_term(const unsigned int                                face_no,
                            const FEFaceValuesBase<dim>&                      fe_v,
                            const FEFaceValuesBase<dim>&                      fe_v_neighbor,
                            const bool                                        external_face,
                            const unsigned int                                boundary_id,
                            const Vector<double>&                             local_current_solution,
                            const Vector<double>&                             local_current_solution_neighbor,
                            const std::vector<types::global_dof_index>&       dof_indices,
                            const std::vector<types::global_dof_index>&       dof_indices_neighbor) const;

    void vmult(LinearAlgebra::distributed::Vector<double>&       dst,
               const LinearAlgebra::distributed::Vector<double>& src) const;

  private:
    const DoFHandler<dim>& dof_handler;
    const MappingQ1<dim>&  mapping;
    const QGauss<dim>&     quadrature;
    const QGauss<dim - 1>& face_quadrature;
    TimerOutput&           time_table;
    Parameters::AllParameters<dim> parameters;  //--- Auxiliary variable to read parameters

    double             lambda_old; //--- Extra parameter to save old Lax-Friedrichs stability (for explicit caching)
    double             gamma_2; //--- Extra parameter of TR_BDF2
    double             gamma_3; //--- Extra parameter of TR_BDF2
    unsigned int       TR_BDF2_stage; //--- Extra parameter for counting stage TR_BDF2

    LA::MPI::Vector&   locally_relevant_solution;

    mutable DiagonalMatrix<LinearAlgebra::distributed::Vector<double>> inverse_diagonal_entries;
  };

  // Class constructor
  template<int dim>
  ConservationLawPreconditioner<dim>::ConservationLawPreconditioner(const DoFHandler<dim>& dof_handler,
                                                                    const MappingQ1<dim>&  mapping,
                                                                    const QGauss<dim>&     quadrature,
                                                                    const QGauss<dim - 1>& face_quadrature,
                                                                    TimerOutput&           time_table,
                                                                    ParameterHandler&      prm,
                                                                    LA::MPI::Vector&       locally_relevant_solution):
    dof_handler(dof_handler), mapping(mapping), quadrature(quadrature), face_quadrature(face_quadrature), time_table(time_table),
    parameters(prm), locally_relevant_solution(locally_relevant_solution) {
      //--- Compute the other two parameters of TR_BDF2 scheme
      if(parameters.time_integration_scheme == "TR_BDF2") {
        parameters.theta = 1.0 - std::sqrt(2)/2.0;
        gamma_2 = (1.0 - 2.0*parameters.theta)/(2.0*(1.0 - parameters.theta));
        gamma_3 = (1.0 - gamma_2)/(2.0*parameters.theta);
        TR_BDF2_stage = 1;
      }
  }

  // Set stage for TR_BDF2 (fundamental because the effectvve value is known only at
  // runtime and so has to be called by ConservationLaw class)
  template<int dim>
  void ConservationLawPreconditioner<dim>::set_TR_BDF2_stage(const unsigned int stage) {
    Assert(stage == 1 || stage == 2, ExcInternalError());
    TR_BDF2_stage = stage;
  }

  // Set time for boundary conditions (fundamental because the effectvve value is known only at
  // runtime and so has to be called by ConservationLaw class)
  template<int dim>
  void ConservationLawPreconditioner<dim>::set_current_time(const double current_time) {
    if(parameters.testcase == 1)
      parameters.exact_solution.set_time(current_time);
  }

  // Set lambda_old for LF (fundamental because the effectvve value is known only at
  // runtime and so has to be called by ConservationLaw class)
  template<int dim>
  void ConservationLawPreconditioner<dim>::set_lambda_old(const double lambda_old) {
    this->lambda_old = lambda_old;
  }

  // Assemble cell term
  template<int dim>
  void ConservationLawPreconditioner<dim>::assemble_cell_term(const FEValues<dim>&                        fe_v,
                                                              const Vector<double>&                       local_current_solution,
                                                              const std::vector<types::global_dof_index>& dof_indices) const {
    TimerOutput::Scope t(time_table, "Assemble preconditioner cell term");

    const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
    const unsigned int n_q_points    = fe_v.n_quadrature_points;

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    cell_matrix = 0;

    Table<2, Sacado::Fad::DFad<double>> W(n_q_points, EulerEquations<dim>::n_components);

    std::vector<Sacado::Fad::DFad<double>> independent_local_dof_values(dofs_per_cell);
    for(unsigned int i = 0; i < dofs_per_cell; ++i) {
      independent_local_dof_values[i] = local_current_solution[i];
      independent_local_dof_values[i].diff(i, dofs_per_cell);
    }

    for(unsigned int q = 0; q < n_q_points; ++q)
      for(unsigned int c = 0; c < EulerEquations<dim>::n_components; ++c)
        W[q][c] = 0;

    for(unsigned int q = 0; q < n_q_points; ++q) {
      for(unsigned int i = 0; i < dofs_per_cell; ++i) {
        const unsigned int c = fe_v.get_fe().system_to_component_index(i).first;
        W[q][c] += independent_local_dof_values[i]*fe_v.shape_value_component(i, q, c);
      }
    }

    std::vector<std::array<std::array<Sacado::Fad::DFad<double>, dim>,
                           EulerEquations<dim>::n_components>>          flux(n_q_points);

    std::vector<std::array<Sacado::Fad::DFad<double>, EulerEquations<dim>::n_components>> forcing(n_q_points);

    for(unsigned int q = 0; q < n_q_points; ++q) {
      EulerEquations<dim>::compute_flux_matrix(W[q], flux[q]);
      EulerEquations<dim>::compute_forcing_vector(W[q], forcing[q], parameters.testcase);
    }


    for(unsigned int i = 0; i < dofs_per_cell; ++i) {
      Sacado::Fad::DFad<double> R_i = 0;

      const unsigned int component_i = fe_v.get_fe().system_to_component_index(i).first;

      for(unsigned int point = 0; point < fe_v.n_quadrature_points; ++point) {
        //--- Reorganize residual computation
        if(parameters.time_integration_scheme == "Theta_Method") {
          if(parameters.is_stationary == false)
            R_i += 1.0/parameters.time_step *
                   W[point][component_i] *
                   fe_v.shape_value_component(i, point, component_i) *
                   fe_v.JxW(point);

          for(unsigned int d = 0; d < dim; d++)
            R_i -= parameters.theta*flux[point][component_i][d]*
                   fe_v.shape_grad_component(i, point, component_i)[d] *
                   fe_v.JxW(point);

          R_i -= parameters.theta*forcing[point][component_i]*
                 fe_v.shape_value_component(i, point, component_i) *
                 fe_v.JxW(point);
        }
        //--- Stages of TR-BDF2
        else {
          //--- First stage of TR_BDF2
          if(TR_BDF2_stage == 1) {
            if(parameters.is_stationary == false)
              R_i += 1.0/parameters.time_step *
                     W[point][component_i] *
                     fe_v.shape_value_component(i, point, component_i) *
                     fe_v.JxW(point);

            for(unsigned int d = 0; d < dim; d++)
              R_i -= parameters.theta*flux[point][component_i][d] *
                     fe_v.shape_grad_component(i, point, component_i)[d] *
                     fe_v.JxW(point);

            R_i -= parameters.theta*forcing[point][component_i] *
                   fe_v.shape_value_component(i, point, component_i) *
                   fe_v.JxW(point);
          }
          //--- Second stage of TR_BDF2
          else if(TR_BDF2_stage == 2) {
            if(parameters.is_stationary == false)
              R_i += 1.0/parameters.time_step *
                     W[point][component_i] *
                     fe_v.shape_value_component(i, point, component_i) *
                     fe_v.JxW(point);

            for(unsigned int d = 0; d < dim; d++)
              R_i -= gamma_2*flux[point][component_i][d] *
                     fe_v.shape_grad_component(i, point, component_i)[d] *
                     fe_v.JxW(point);

            R_i -=  gamma_2*forcing[point][component_i] *
                    fe_v.shape_value_component(i, point, component_i) *
                    fe_v.JxW(point);
          }
        }
      }
      inverse_diagonal_entries.add(dof_indices[i], dof_indices[i], 1.0/(R_i.fastAccessDx(0)));
    }
  }

  // Assemble face term
  template<int dim>
  void ConservationLawPreconditioner<dim>::assemble_face_term(const unsigned int                          face_no,
                                                              const FEFaceValuesBase<dim>&                fe_v,
                                                              const FEFaceValuesBase<dim>&                fe_v_neighbor,
                                                              const bool                                  external_face,
                                                              const unsigned int                          boundary_id,
                                                              const Vector<double>&                       local_current_solution,
                                                              const Vector<double>&                       local_current_solution_neighbor,
                                                              const std::vector<types::global_dof_index>& dof_indices,
                                                              const std::vector<types::global_dof_index>& dof_indices_neighbor) const {
    TimerOutput::Scope t(time_table, "Assemble preconditioner face term");

    const unsigned int n_q_points    = fe_v.n_quadrature_points;
    const unsigned int dofs_per_cell = fe_v.dofs_per_cell;

    std::vector<Sacado::Fad::DFad<double>> independent_local_dof_values(dofs_per_cell),
                                           independent_neighbor_dof_values(external_face == false ?
                                                                           dofs_per_cell : 0);

    const unsigned int n_independent_variables = (external_face == false ? dofs_per_cell + dofs_per_cell : dofs_per_cell);

    for(unsigned int i = 0; i < dofs_per_cell; i++) {
      independent_local_dof_values[i] = local_current_solution[i];
      independent_local_dof_values[i].diff(i, n_independent_variables);
    }

    if(external_face == false) {
      for(unsigned int i = 0; i < dofs_per_cell; i++) {
        independent_neighbor_dof_values[i] = local_current_solution_neighbor[i];
        independent_neighbor_dof_values[i].diff(i + dofs_per_cell, n_independent_variables);
      }
    }

    Table<2, Sacado::Fad::DFad<double>> Wplus(n_q_points, EulerEquations<dim>::n_components),
                                        Wminus(n_q_points, EulerEquations<dim>::n_components);

    for(unsigned int q = 0; q < n_q_points; ++q) {
      for(unsigned int i = 0; i < dofs_per_cell; ++i) {
        const unsigned int component_i = fe_v.get_fe().system_to_component_index(i).first;
        Wplus[q][component_i] += independent_local_dof_values[i] *
                                 fe_v.shape_value_component(i, q, component_i);
      }
    }

    // Computing "opposite side" is a bit more complicated. If this is
    // an internal face, we can compute it as above by simply using the
    // independent variables from the neighbor:
    if(external_face == false) {
      for(unsigned int q = 0; q < n_q_points; ++q) {
        for(unsigned int i = 0; i < dofs_per_cell; ++i) {
          const unsigned int component_i = fe_v_neighbor.get_fe().system_to_component_index(i).first;
          Wminus[q][component_i] += independent_neighbor_dof_values[i] *
                                    fe_v_neighbor.shape_value_component(i, q, component_i);
        }
      }
    }
    else {
      Assert(boundary_id < Parameters::AllParameters<dim>::max_n_boundaries,
             ExcIndexRange(boundary_id, 0, Parameters::AllParameters<dim>::max_n_boundaries));

      std::vector<Vector<double>> boundary_values(n_q_points, Vector<double>(EulerEquations<dim>::n_components));
      if(parameters.testcase == 0)
        parameters.boundary_conditions[boundary_id].values.vector_value_list(fe_v.get_quadrature_points(), boundary_values);
      else
        parameters.exact_solution.vector_value_list(fe_v.get_quadrature_points(), boundary_values);

      for(unsigned int q = 0; q < n_q_points; q++) {
        // Here we assume that boundary type, boundary normal vector and
        // boundary data values maintain the same during time advancing.
        EulerEquations<dim>::compute_Wminus(parameters.boundary_conditions[boundary_id].kind,
                                            fe_v.normal_vector(q), Wplus[q], boundary_values[q], Wminus[q]);
      }
    }

    std::vector<std::array<Sacado::Fad::DFad<double>, EulerEquations<dim>::n_components>> normal_fluxes(n_q_points);

    for(unsigned int q = 0; q < n_q_points; ++q)
      EulerEquations<dim>::numerical_normal_flux(fe_v.normal_vector(q), Wplus[q], Wminus[q], lambda_old, normal_fluxes[q]);

    for(unsigned int i = 0; i < dofs_per_cell; ++i) {
      if(fe_v.get_fe().has_support_on_face(i, face_no) == true) {
        Sacado::Fad::DFad<double> R_i = 0;

        for(unsigned int point = 0; point < n_q_points; ++point) {
          const unsigned int component_i = fe_v.get_fe().system_to_component_index(i).first;

          //--- Reorganizing contributions
          if(parameters.time_integration_scheme == "Theta_Method") {
            R_i += parameters.theta*normal_fluxes[point][component_i]*
                   fe_v.shape_value_component(i, point, component_i) *
                   fe_v.JxW(point);
          }
          //--- TR_BDF2 assembling
          else {
            //--- First stage of TR_BDF2
            if(TR_BDF2_stage == 1) {
              R_i += parameters.theta*normal_fluxes[point][component_i] *
                     fe_v.shape_value_component(i, point, component_i) *
                     fe_v.JxW(point);
            }
            //--- Second stage of TR_BDF2
            else if(TR_BDF2_stage == 2) {
              R_i += gamma_2*normal_fluxes[point][component_i]*
                     fe_v.shape_value_component(i, point, component_i) *
                     fe_v.JxW(point);
            }
          }
        }
        inverse_diagonal_entries.add(dof_indices[i], dof_indices[i], 1.0/(R_i.fastAccessDx(0)));
        if(external_face == false)
          inverse_diagonal_entries.add(dof_indices_neighbor[i], dof_indices_neighbor[i], 1.0/(R_i.fastAccessDx(dofs_per_cell)));
      }
    }
  }

  // Now we compute the action of the operator, namely the matrix-vector operator
  template<int dim>
  void ConservationLawPreconditioner<dim>::vmult(LinearAlgebra::distributed::Vector<double>&       dst,
                                                 const LinearAlgebra::distributed::Vector<double>& src) const {
    TimerOutput::Scope t(time_table, "Action of linear operator preconditioner");

    const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices_neighbor(dofs_per_cell);

    dst = 0;
    inverse_diagonal_entries.reinit(dst);
    src.update_ghost_values();

    const UpdateFlags update_flags = update_values | update_gradients |
                                     update_quadrature_points | update_JxW_values,
                      face_update_flags = update_values | update_quadrature_points |
                                          update_JxW_values |
                                          update_normal_vectors,
                      neighbor_face_update_flags = update_values;

    FEValues<dim>        fe_v(mapping, dof_handler.get_fe(), quadrature, update_flags);
    FEFaceValues<dim>    fe_v_face(mapping, dof_handler.get_fe(), face_quadrature, face_update_flags);
    FESubfaceValues<dim> fe_v_subface(mapping, dof_handler.get_fe(), face_quadrature, face_update_flags);
    FEFaceValues<dim>    fe_v_face_neighbor(mapping, dof_handler.get_fe(), face_quadrature, neighbor_face_update_flags);
    FESubfaceValues<dim> fe_v_subface_neighbor(mapping, dof_handler.get_fe(), face_quadrature, neighbor_face_update_flags);

    Vector<double> local_current_solution(dofs_per_cell),
                   local_current_solution_neighbor(dofs_per_cell);

    // Then loop over all cells, initialize the FEValues object for the
    // current cell and call the function that assembles the problem on this
    // cell.
    for(const auto& cell: dof_handler.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        fe_v.reinit(cell);
        cell->get_dof_values(locally_relevant_solution, local_current_solution);
        cell->get_dof_indices(dof_indices);
        assemble_cell_term(fe_v, local_current_solution, dof_indices);

        // Then loop over all the faces of this cell.  If a face is part of
        // the external boundary, then assemble boundary conditions there:
        for(unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no) {
          if(cell->at_boundary(face_no)) {
            fe_v_face.reinit(cell, face_no);
            assemble_face_term(face_no, fe_v_face, fe_v_face,
                               true, cell->face(face_no)->boundary_id(),
                               local_current_solution, local_current_solution,
                               dof_indices, dof_indices);
          }

          // The alternative is that we are dealing with an internal face.
          else {
            if(cell->neighbor(face_no)->has_children()) {
              const unsigned int neighbor2 = cell->neighbor_of_neighbor(face_no);

              for(unsigned int subface_no = 0; subface_no < cell->face(face_no)->n_children(); ++subface_no) {
                const typename DoFHandler<dim>::active_cell_iterator
                neighbor_child = cell->neighbor_child_on_subface(face_no, subface_no);

                Assert(neighbor_child->face(neighbor2) == cell->face(face_no)->child(subface_no), ExcInternalError());
                Assert(neighbor_child->has_children() == false, ExcInternalError());

                fe_v_subface.reinit(cell, face_no, subface_no);
                fe_v_face_neighbor.reinit(neighbor_child, neighbor2);
                neighbor_child->get_dof_values(locally_relevant_solution, local_current_solution_neighbor);
                neighbor_child->get_dof_indices(dof_indices_neighbor);

                assemble_face_term(face_no, fe_v_subface, fe_v_face_neighbor,
                                   false, numbers::invalid_unsigned_int,
                                   local_current_solution, local_current_solution_neighbor,
                                   dof_indices, dof_indices_neighbor);
              }
            }

            // The other possibility we have to care for is if the neighbor
            // is coarser than the current cell:
            else if(cell->neighbor(face_no)->level() != cell->level()) {
              const typename DoFHandler<dim>::cell_iterator neighbor = cell->neighbor(face_no);
              Assert(neighbor->level() == cell->level() - 1, ExcInternalError());

              const std::pair<unsigned int, unsigned int> faceno_subfaceno = cell->neighbor_of_coarser_neighbor(face_no);
              const unsigned int neighbor_face_no = faceno_subfaceno.first,
                                 neighbor_subface_no = faceno_subfaceno.second;

              Assert(neighbor->neighbor_child_on_subface(neighbor_face_no, neighbor_subface_no) == cell,
                     ExcInternalError());

              fe_v_face.reinit(cell, face_no);
              fe_v_subface_neighbor.reinit(neighbor, neighbor_face_no, neighbor_subface_no);
              neighbor->get_dof_values(locally_relevant_solution, local_current_solution_neighbor);
              neighbor->get_dof_indices(dof_indices_neighbor);

              assemble_face_term(face_no, fe_v_face, fe_v_subface_neighbor,
                                 false, numbers::invalid_unsigned_int,
                                 local_current_solution, local_current_solution_neighbor,
                                 dof_indices, dof_indices_neighbor);
            }
            // Same refinement level
            else {
              fe_v_face.reinit(cell, face_no);
              const typename DoFHandler<dim>::cell_iterator neighbor = cell->neighbor(face_no);
              const unsigned int other_face_no = cell->neighbor_of_neighbor(face_no);
              fe_v_face_neighbor.reinit(neighbor, other_face_no);
              neighbor->get_dof_values(locally_relevant_solution, local_current_solution_neighbor);
              neighbor->get_dof_indices(dof_indices_neighbor);
              assemble_face_term(face_no, fe_v_face, fe_v_face_neighbor,
                                 false, numbers::invalid_unsigned_int,
                                 local_current_solution, local_current_solution_neighbor,
                                 dof_indices, dof_indices_neighbor);
            }
          }
        }
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    inverse_diagonal_entries.vmult(dst, src);
  }


  // @sect3{Conservation law class}

  // Here finally comes the class that actually does something with all the
  // Euler equation and parameter specifics we've defined above. The public
  // interface is pretty much the same as always (the constructor now takes
  // the name of a file from which to read parameters, which is passed on the
  // command line). The private function interface is also pretty similar to
  // the usual arrangement, with the <code>assemble_system</code> function
  // split into three parts: one that contains the main loop over all cells
  // and that then calls the other two for integrals over cells and faces,
  // respectively.
  template<int dim>
  class ConservationLaw {
  public:
    ConservationLaw(ParameterHandler& prm);
    void run();

  private:
    //--- Function to build the grid and distribute dofs
    void make_grid_and_dofs();

    void setup_system();

    void assemble_explicit_cell_term(const FEValues<dim>& fe_v);
    void assemble_explicit_face_term(const unsigned int           face_no,
                                     const FEFaceValuesBase<dim>& fe_v,
                                     const FEFaceValuesBase<dim>& fe_v_neighbor,
                                     const bool                   external_face,
                                     const unsigned int           boundary_id);
    void copy_local_to_global_explicit(const std::vector<types::global_dof_index>& dof_indices);
    void assemble_explicit_system();

    void assemble_rhs_cell_term(const FEValues<dim>& fe_v);
    void assemble_rhs_face_term(const unsigned int           face_no,
                                const FEFaceValuesBase<dim>& fe_v,
                                const FEFaceValuesBase<dim>& fe_v_neighbor,
                                const bool                   external_face,
                                const unsigned int           boundary_id);
    void copy_local_to_global(const std::vector<types::global_dof_index>& dof_indices);
    void assemble_rhs();

    std::pair<unsigned int, double> solve(LinearAlgebra::distributed::Vector<double>& newton_update);

    void compute_refinement_indicators(Vector<double>& indicator) const;
    void refine_grid(const Vector<double>& indicator);

    void compute_errors();

    void output_results() const;


    Parameters::AllParameters<dim> parameters;  //--- Auxiliary variable to read parameters

    // The first few member variables are also rather standard. Note that we
    // define a mapping object to be used throughout the program when
    // assembling terms (we will hand it to every FEValues and FEFaceValues
    // object); the mapping we use is just the standard $Q_1$ mapping --
    // nothing fancy, in other words -- but declaring one here and using it
    // throughout the program will make it simpler later on to change it if
    // that should become necessary. This is, in fact, rather pertinent: it is
    // known that for transsonic simulations with the Euler equations,
    // computations do not converge even as $h\rightarrow 0$ if the boundary
    // approximation is not of sufficiently high order.
    MPI_Comm communicator; //--- Communicator for parallel computations

    parallel::distributed::Triangulation<dim> triangulation;
    const MappingQ1<dim> mapping;

    FESystem<dim>   fe;
    DoFHandler<dim> dof_handler;

    //--- Figure out who are my dofs and my locally relevant dofs
    IndexSet locally_relevant_dofs;
    IndexSet locally_owned_dofs;

    AffineConstraints<double> constraints; //--- Empty variable for distribute local_to_global

    QGauss<dim>     quadrature;
    QGauss<dim - 1> face_quadrature;

    // Next come a number of data vectors that correspond to the solution of
    // the previous time step (<code>old_solution</code>), the best guess of
    // the current solution (<code>current_solution</code>; we say
    // <i>guess</i> because the Newton iteration to compute it may not have
    // converged yet, whereas <code>old_solution</code> refers to the fully
    // converged final result of the previous time step), and a predictor for
    // the solution at the next time step, computed by extrapolating the
    // current and previous solution one time step into the future:
    LA::MPI::Vector old_solution;
    LA::MPI::Vector current_solution;
    LA::MPI::Vector intermediate_solution; //--- Extra variable for TR_BDF2
    LA::MPI::Vector predictor;

    LA::MPI::Vector right_hand_side;
    LA::MPI::Vector right_hand_side_explicit;

    LA::MPI::Vector locally_relevant_solution,
                    locally_relevant_old_solution;  //--- Extra variables for parallel purposes (read-only)

    LinearAlgebra::distributed::Vector<double> right_hand_side_mf;
    LinearAlgebra::ReadWriteVector<double>     right_hand_side_exchanger;

    // This final set of member variables (except for the object holding all
    // run-time parameters at the very bottom and a screen output stream that
    // only prints something if verbose output has been requested) deals with
    // the interface we have in this program to the Trilinos library that
    // provides us with linear solvers. Similarly to including PETSc matrices
    // in step-17, step-18, and step-19, all we need to do is to create a
    // Trilinos sparse matrix instead of the standard deal.II class. The
    // system matrix is used for the Jacobian in each Newton step. Since we do
    // not intend to run this program in parallel (which wouldn't be too hard
    // with Trilinos data structures, though), we don't have to think about
    // anything else like distributing the degrees of freedom.

    Vector<double>     cell_rhs;    //--- Local rhs
    Vector<double>     local_current_solution,
                       local_intermediate_solution,
                       local_old_solution,
                       local_current_solution_neighbor,
                       local_old_solution_neighbor;

    double             lambda_old; //--- Extra parameter to save old Lax-Friedrichs stability (for explicit caching)
    double             gamma_2; //--- Extra parameter of TR_BDF2
    double             gamma_3; //--- Extra parameter of TR_BDF2
    unsigned int       TR_BDF2_stage; //--- Extra parameter for counting stage TR_BDF2
    ConditionalOStream pcout; //--- Extra parameter for parallel cout
    ConditionalOStream verbose_cout;
    std::ofstream      output_niter;
    std::ofstream      output_error;
    std::ofstream      time_out;     //--- Auxiliary ofstream for time output
    ConditionalOStream ptime_out;    //--- Auxiliary conditional stream for time output
    TimerOutput        time_table;   //--- Auxiliary Table for time

    ConservationLawOperator<dim>       matrix_free;
    ConservationLawPreconditioner<dim> preconditioner;
  };


  // @sect4{ConservationLaw::ConservationLaw}
  //
  // There is nothing much to say about the constructor. Essentially, it reads
  // the input file and fills the parameter object with the parsed values:
  template<int dim>
  ConservationLaw<dim>::ConservationLaw(ParameterHandler& prm):
      parameters(prm),
      communicator(MPI_COMM_WORLD),
      triangulation(communicator),
      mapping(),
      fe(FE_DGQ<dim>(parameters.fe_degree), EulerEquations<dim>::n_components),
      dof_handler(triangulation),
      quadrature(fe.degree + 1),
      face_quadrature(fe.degree + 1),
      cell_rhs(fe.dofs_per_cell),
      local_current_solution(fe.dofs_per_cell),
      local_intermediate_solution(fe.dofs_per_cell),
      local_old_solution(fe.dofs_per_cell),
      local_current_solution_neighbor(fe.dofs_per_cell),
      local_old_solution_neighbor(fe.dofs_per_cell),
      pcout(std::cout, Utilities::MPI::this_mpi_process(communicator) == 0),
      verbose_cout(std::cout, Utilities::MPI::this_mpi_process(communicator) == 0 &&
                              parameters.output == Parameters::Solver::verbose),
      output_niter("./" + parameters.dir + "/linear_number_iterations.dat", std::ofstream::out),
      output_error("./" + parameters.dir + "/error_analysis.dat", std::ofstream::out),
      time_out("./" + parameters.dir + "/time_analysis_" +
               Utilities::int_to_string(Utilities::MPI::n_mpi_processes(communicator)) + "proc.dat"),
      ptime_out(time_out, Utilities::MPI::this_mpi_process(communicator) == 0),
      time_table(ptime_out, TimerOutput::never, TimerOutput::wall_times),
      matrix_free(dof_handler, mapping, quadrature, face_quadrature, time_table, prm, locally_relevant_solution),
      preconditioner(dof_handler, mapping, quadrature, face_quadrature, time_table, prm, locally_relevant_solution) {
    //--- Compute the other two parameters of TR_BDF2 scheme
    if(parameters.time_integration_scheme == "TR_BDF2") {
      parameters.theta = 1.0 - std::sqrt(2)/2.0;
      gamma_2 = (1.0 - 2.0*parameters.theta)/(2.0*(1.0 - parameters.theta));
      gamma_3 = (1.0 - gamma_2)/(2.0*parameters.theta);
      TR_BDF2_stage = 1;
    }
  }

  // @sect4{ConservationLaw::make_grid_and_dofs}
  //
  // The following (easy) function is called at the beginnig and sets grid and
  // dofs according to the testcase. Finally it initializes the various vectors for
  // solution, predictor and rhs.
  template<int dim>
  void ConservationLaw<dim>::make_grid_and_dofs() {
    TimerOutput::Scope t(time_table, "Make grid");

    if(parameters.testcase == 0) {
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(triangulation);

      std::ifstream input_file(parameters.mesh_filename);
      Assert(input_file, ExcFileNotOpen(parameters.mesh_filename.c_str()));

      grid_in.read_ucd(input_file);
    }
    else {
      Point<dim> lower_left;
      for(unsigned int d = 1; d < dim; ++d)
        lower_left[d] = -5;

      Point<dim> upper_right;
      upper_right[0] = 10;
      for(unsigned int d = 1; d < dim; ++d)
        upper_right[d] = 5;

      GridGenerator::hyper_rectangle(triangulation, lower_left, upper_right);
      triangulation.refine_global(parameters.global_refinements);
    }
    dof_handler.clear();
    dof_handler.distribute_dofs(fe);

    //Extract locally dofs
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    //Size of all fields
    old_solution.reinit(locally_owned_dofs, communicator);
    current_solution.reinit(locally_owned_dofs, communicator);
    predictor.reinit(locally_owned_dofs, communicator);
    right_hand_side.reinit(locally_owned_dofs, communicator);
    right_hand_side_explicit.reinit(locally_owned_dofs, communicator);
  }


  // @sect4{ConservationLaw::setup_system}
  //
  // The following (easy) function is called each time the mesh is
  // changed. All it does is to resize the Trilinos matrix according to a
  // sparsity pattern that we generate as in all the previous tutorial
  // programs.
  template<int dim>
  void ConservationLaw<dim>::setup_system() {
    TimerOutput::Scope t(time_table, "Setup system");

    right_hand_side_mf.reinit(locally_owned_dofs, communicator);
    right_hand_side_exchanger.reinit(locally_owned_dofs, communicator);

    //--- Read-only variant of the solution that must be set after the solution
    locally_relevant_solution.reinit(locally_owned_dofs, locally_relevant_dofs, communicator);
    locally_relevant_old_solution.reinit(locally_owned_dofs, locally_relevant_dofs, communicator);
  }


  // @sect4{ConservationLaw::assemble_explicit_cell_term}
  //
  template<int dim>
  void ConservationLaw<dim>::assemble_explicit_cell_term(const FEValues<dim>& fe_v) {
    TimerOutput::Scope t(time_table, "Assemble explicit cell term");

    const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
    const unsigned int n_q_points    = fe_v.n_quadrature_points;

    cell_rhs = 0;

    Table<2, double> W_old(n_q_points, EulerEquations<dim>::n_components);

    //--- Intermediate flux value
    Table<2, double> W_int(n_q_points, EulerEquations<dim>::n_components);

    for(unsigned int q = 0; q < n_q_points; ++q) {
      for(unsigned int c = 0; c < EulerEquations<dim>::n_components; ++c) {
        W_old[q][c] = 0;
        //--- Setting to zero flux second stage TR_BDF2
        if(parameters.time_integration_scheme == "TR_BDF2" && TR_BDF2_stage == 2)
          W_int[q][c] = 0;
      }
    }

    for(unsigned int q = 0; q < n_q_points; ++q) {
      for(unsigned int i = 0; i < dofs_per_cell; ++i) {
        const unsigned int c = fe_v.get_fe().system_to_component_index(i).first;

        W_old[q][c] += local_old_solution[i]*fe_v.shape_value_component(i, q, c);

        //--- Setting W_int for second stage TR_BDF2
        if(parameters.time_integration_scheme == "TR_BDF2" && TR_BDF2_stage == 2)
          W_int[q][c] += local_intermediate_solution[i]*fe_v.shape_value_component(i, q, c);
      }
    }

    std::vector<std::array<std::array<double, dim>, EulerEquations<dim>::n_components>> flux_old(n_q_points);
    std::vector<std::array<double, EulerEquations<dim>::n_components>> forcing_old(n_q_points);

    for(unsigned int q = 0; q < n_q_points; ++q) {
      EulerEquations<dim>::compute_flux_matrix(W_old[q], flux_old[q]);
      EulerEquations<dim>::compute_forcing_vector(W_old[q], forcing_old[q], parameters.testcase);
    }

    for(unsigned int i = 0; i < dofs_per_cell; ++i) {
      double R_i = 0;

      const unsigned int component_i = fe_v.get_fe().system_to_component_index(i).first;

      for(unsigned int point = 0; point < fe_v.n_quadrature_points; ++point) {
        //--- Reorganize residual computation
        if(parameters.time_integration_scheme == "Theta_Method") {
          if(parameters.is_stationary == false)
            R_i -= 1.0/parameters.time_step *
                   W_old[point][component_i] *
                   fe_v.shape_value_component(i, point, component_i) *
                   fe_v.JxW(point);

          for(unsigned int d = 0; d < dim; d++)
            R_i -= (1.0 - parameters.theta)*flux_old[point][component_i][d]*
                   fe_v.shape_grad_component(i, point, component_i)[d] *
                   fe_v.JxW(point);

          R_i -= (1.0 - parameters.theta)*forcing_old[point][component_i]*
                 fe_v.shape_value_component(i, point, component_i) *
                 fe_v.JxW(point);
        }
        //--- Stages of TR-BDF2
        else {
          //--- First stage of TR_BDF2
          if(TR_BDF2_stage == 1) {
            if(parameters.is_stationary == false)
              R_i -= 1.0/parameters.time_step *
                     W_old[point][component_i] *
                     fe_v.shape_value_component(i, point, component_i) *
                     fe_v.JxW(point);

            for(unsigned int d = 0; d < dim; d++)
              R_i -= parameters.theta*flux_old[point][component_i][d] *
                     fe_v.shape_grad_component(i, point, component_i)[d] *
                     fe_v.JxW(point);

            R_i -= parameters.theta*forcing_old[point][component_i] *
                   fe_v.shape_value_component(i, point, component_i) *
                   fe_v.JxW(point);
          }
          //--- Second stage of TR_BDF2
          else if(TR_BDF2_stage == 2) {
            if(parameters.is_stationary == false)
              R_i -= 1.0/parameters.time_step *
                     (gamma_3*W_int[point][component_i] + (1.0 - gamma_3)*W_old[point][component_i])*
                     fe_v.shape_value_component(i, point, component_i) *
                     fe_v.JxW(point);
          }
        }

      }

      cell_rhs(i) -= R_i;
    }
  }


  // @sect4{ConservationLaw::assemble_explicit_face_term}
  //
  template<int dim>
  void ConservationLaw<dim>::assemble_explicit_face_term(const unsigned int           face_no,
                                                         const FEFaceValuesBase<dim>& fe_v,
                                                         const FEFaceValuesBase<dim>& fe_v_neighbor,
                                                         const bool                   external_face,
                                                         const unsigned int           boundary_id) {
    TimerOutput::Scope t(time_table, "Assemble explicit face term");

    cell_rhs = 0;

    const unsigned int n_q_points    = fe_v.n_quadrature_points;
    const unsigned int dofs_per_cell = fe_v.dofs_per_cell;

    Table<2, double> Wplus_old(n_q_points,  EulerEquations<dim>::n_components),
                     Wminus_old(n_q_points, EulerEquations<dim>::n_components);

    for(unsigned int q = 0; q < n_q_points; ++q) {
      for(unsigned int i = 0; i < dofs_per_cell; ++i) {
        const unsigned int component_i = fe_v.get_fe().system_to_component_index(i).first;
        Wplus_old[q][component_i] += local_old_solution[i] *
                                     fe_v.shape_value_component(i, q, component_i);
      }
    }

    // Computing "opposite side" is a bit more complicated. If this is
    // an internal face, we can compute it as above by simply using the
    // independent variables from the neighbor:
    if(external_face == false) {
      for(unsigned int q = 0; q < n_q_points; ++q) {
        for(unsigned int i = 0; i < dofs_per_cell; ++i) {
          const unsigned int component_i = fe_v_neighbor.get_fe().system_to_component_index(i).first;
          Wminus_old[q][component_i] += local_old_solution_neighbor[i] *
                                        fe_v_neighbor.shape_value_component(i, q, component_i);
        }
      }
    }
    else {
      Assert(boundary_id < Parameters::AllParameters<dim>::max_n_boundaries,
             ExcIndexRange(boundary_id, 0, Parameters::AllParameters<dim>::max_n_boundaries));

      std::vector<Vector<double>> boundary_values(n_q_points, Vector<double>(EulerEquations<dim>::n_components));
      if(parameters.testcase == 0)
        parameters.boundary_conditions[boundary_id].values.vector_value_list(fe_v.get_quadrature_points(), boundary_values);
      else
        parameters.exact_solution.vector_value_list(fe_v.get_quadrature_points(), boundary_values);

      for(unsigned int q = 0; q < n_q_points; q++)
        EulerEquations<dim>::compute_Wminus(parameters.boundary_conditions[boundary_id].kind,
                                            fe_v.normal_vector(q), Wplus_old[q], boundary_values[q], Wminus_old[q]);
    }

    std::vector<std::array<double, EulerEquations<dim>::n_components>> normal_fluxes_old(n_q_points);

    for(unsigned int q = 0; q < n_q_points; ++q) {
      lambda_old = std::max(EulerEquations<dim>::compute_velocity(Wplus_old[q])  +
                            EulerEquations<dim>::compute_speed_of_sound(Wplus_old[q]),
                            EulerEquations<dim>::compute_velocity(Wminus_old[q]) +
                            EulerEquations<dim>::compute_speed_of_sound(Wminus_old[q]));
      EulerEquations<dim>::numerical_normal_flux(fe_v.normal_vector(q), Wplus_old[q], Wminus_old[q], lambda_old, normal_fluxes_old[q]);
    }

    for(unsigned int i = 0; i < dofs_per_cell; ++i) {
      if(fe_v.get_fe().has_support_on_face(i, face_no) == true) {
        double R_i = 0.0;

        for(unsigned int point = 0; point < n_q_points; ++point) {
          const unsigned int component_i = fe_v.get_fe().system_to_component_index(i).first;

          //--- Reorganizing contributions
          if(parameters.time_integration_scheme == "Theta_Method") {
            R_i += (1.0 - parameters.theta)*normal_fluxes_old[point][component_i]*
                   fe_v.shape_value_component(i, point, component_i) *
                   fe_v.JxW(point);
          }
          //--- TR_BDF2 assembling
          else {
            //--- First stage of TR_BDF2
            if(TR_BDF2_stage == 1) {
              R_i += parameters.theta*normal_fluxes_old[point][component_i] *
                     fe_v.shape_value_component(i, point, component_i) *
                     fe_v.JxW(point);
            }
          }
        }

        cell_rhs(i) -= R_i;
      }
    }
  }


  // @sect4{ConservationLaw::copy_local_to_global_explicit}
  //
  template<int dim>
  void ConservationLaw<dim>::copy_local_to_global_explicit(const std::vector<types::global_dof_index>& dof_indices) {
    TimerOutput::Scope t(time_table, "Copy local to global explicit");

    constraints.distribute_local_to_global(cell_rhs, dof_indices, right_hand_side_explicit);
    right_hand_side_explicit.compress(VectorOperation::add);
  }


  // @sect4{ConservationLaw::assemble_explicit_system}
  //
  template<int dim>
  void ConservationLaw<dim>::assemble_explicit_system() {
    TimerOutput::Scope t(time_table, "Assemble explicit term");

    right_hand_side_explicit = 0;

    const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    const UpdateFlags update_flags = update_values | update_gradients |
                                     update_quadrature_points | update_JxW_values,
                      face_update_flags = update_values | update_quadrature_points |
                                          update_JxW_values | update_normal_vectors,
                      neighbor_face_update_flags = update_values;

    FEValues<dim> fe_v(mapping, fe, quadrature, update_flags);
    FEFaceValues<dim>    fe_v_face(mapping, fe, face_quadrature, face_update_flags);
    FESubfaceValues<dim> fe_v_subface(mapping, fe, face_quadrature, face_update_flags);
    FEFaceValues<dim>    fe_v_face_neighbor(mapping, fe, face_quadrature, neighbor_face_update_flags);
    FESubfaceValues<dim> fe_v_subface_neighbor(mapping, fe, face_quadrature, neighbor_face_update_flags);

    // Then loop over all cells, initialize the FEValues object for the
    // current cell and call the function that assembles the problem on this
    // cell.
    for(const auto& cell: dof_handler.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        fe_v.reinit(cell);
        cell->get_dof_values(old_solution, local_old_solution);
        if(parameters.time_integration_scheme == "TR_BDF2" && TR_BDF2_stage == 2)
          cell->get_dof_values(intermediate_solution, local_intermediate_solution);
        cell->get_dof_indices(dof_indices);

        //const auto cell_id = std::stoi(cell->id().to_string().substr(4,8));
        assemble_explicit_cell_term(fe_v);
        copy_local_to_global_explicit(dof_indices);

        // Then loop over all the faces of this cell.  If a face is part of
        // the external boundary, then assemble boundary conditions there:
        for(unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no) {
          if(cell->at_boundary(face_no)) {
            fe_v_face.reinit(cell, face_no);
            assemble_explicit_face_term(face_no, fe_v_face, fe_v_face,
                                        true, cell->face(face_no)->boundary_id());
            copy_local_to_global_explicit(dof_indices);
          }

          // The alternative is that we are dealing with an internal face.
          else {
            if(cell->neighbor(face_no)->has_children()) {
              const unsigned int neighbor2 = cell->neighbor_of_neighbor(face_no);

              for(unsigned int subface_no = 0; subface_no < cell->face(face_no)->n_children(); ++subface_no) {
                const typename DoFHandler<dim>::active_cell_iterator
                neighbor_child = cell->neighbor_child_on_subface(face_no, subface_no);

                Assert(neighbor_child->face(neighbor2) == cell->face(face_no)->child(subface_no), ExcInternalError());
                Assert(neighbor_child->has_children() == false, ExcInternalError());

                fe_v_subface.reinit(cell, face_no, subface_no);
                fe_v_face_neighbor.reinit(neighbor_child, neighbor2);
                neighbor_child->get_dof_values(locally_relevant_old_solution, local_old_solution_neighbor);

                assemble_explicit_face_term(face_no, fe_v_subface, fe_v_face_neighbor,
                                            false, numbers::invalid_unsigned_int);
                copy_local_to_global_explicit(dof_indices);
              }
            }

            // The other possibility we have to care for is if the neighbor
            // is coarser than the current cell:
            else if(cell->neighbor(face_no)->level() != cell->level()) {
              const typename DoFHandler<dim>::cell_iterator neighbor = cell->neighbor(face_no);
              Assert(neighbor->level() == cell->level() - 1, ExcInternalError());

              const std::pair<unsigned int, unsigned int> faceno_subfaceno = cell->neighbor_of_coarser_neighbor(face_no);
              const unsigned int neighbor_face_no = faceno_subfaceno.first,
                                 neighbor_subface_no = faceno_subfaceno.second;

              Assert(neighbor->neighbor_child_on_subface(neighbor_face_no, neighbor_subface_no) == cell,
                     ExcInternalError());

              fe_v_face.reinit(cell, face_no);
              fe_v_subface_neighbor.reinit(neighbor, neighbor_face_no,
                                           neighbor_subface_no);
              neighbor->get_dof_values(locally_relevant_old_solution, local_old_solution_neighbor);

              assemble_explicit_face_term(face_no, fe_v_face, fe_v_subface_neighbor,
                                          false, numbers::invalid_unsigned_int);
              copy_local_to_global_explicit(dof_indices);
            }
            // Same refinement level
            else {
              fe_v_face.reinit(cell, face_no);
              const typename DoFHandler<dim>::cell_iterator neighbor = cell->neighbor(face_no);
              const unsigned int other_face_no = cell->neighbor_of_neighbor(face_no);
              fe_v_face_neighbor.reinit(neighbor, other_face_no);
              neighbor->get_dof_values(locally_relevant_old_solution, local_old_solution_neighbor);

              assemble_explicit_face_term(face_no, fe_v_face, fe_v_face_neighbor,
                                          false, numbers::invalid_unsigned_int);
              copy_local_to_global_explicit(dof_indices);
            }
          }
        }
      }
    }
  }


  // @sect4{ConservationLaw::assemble_rhs_cell_term}
  //
  template<int dim>
  void ConservationLaw<dim>::assemble_rhs_cell_term(const FEValues<dim>& fe_v) {
    TimerOutput::Scope t(time_table, "Assemble rhs cell term");

    const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
    const unsigned int n_q_points    = fe_v.n_quadrature_points;

    cell_rhs = 0;

    Table<2, double> W(n_q_points, EulerEquations<dim>::n_components);

    for(unsigned int q = 0; q < n_q_points; ++q)
      for(unsigned int c = 0; c < EulerEquations<dim>::n_components; ++c)
        W[q][c] = 0;

    for(unsigned int q = 0; q < n_q_points; ++q) {
      for(unsigned int i = 0; i < dofs_per_cell; ++i) {
        const unsigned int c = fe_v.get_fe().system_to_component_index(i).first;
        W[q][c] += local_current_solution[i]*fe_v.shape_value_component(i, q, c);
      }
    }

    std::vector<std::array<std::array<double, dim>, EulerEquations<dim>::n_components>>   flux(n_q_points);
    std::vector<std::array<double, EulerEquations<dim>::n_components>>                    forcing(n_q_points);

    for(unsigned int q = 0; q < n_q_points; ++q) {
      EulerEquations<dim>::compute_flux_matrix(W[q], flux[q]);
      EulerEquations<dim>::compute_forcing_vector(W[q], forcing[q], parameters.testcase);
    }

    for(unsigned int i = 0; i < dofs_per_cell; ++i) {
      double R_i = 0.0;

      const unsigned int component_i = fe_v.get_fe().system_to_component_index(i).first;

      for(unsigned int point = 0; point < fe_v.n_quadrature_points; ++point) {
        //--- Reorganize residual computation
        if(parameters.time_integration_scheme == "Theta_Method") {
          if(parameters.is_stationary == false)
            R_i += 1.0/parameters.time_step *
                   W[point][component_i] *
                   fe_v.shape_value_component(i, point, component_i) *
                   fe_v.JxW(point);

          for(unsigned int d = 0; d < dim; d++)
            R_i -= parameters.theta*flux[point][component_i][d]*
                   fe_v.shape_grad_component(i, point, component_i)[d] *
                   fe_v.JxW(point);

          R_i -= parameters.theta*forcing[point][component_i]*
                 fe_v.shape_value_component(i, point, component_i) *
                 fe_v.JxW(point);
        }
        //--- Stages of TR-BDF2
        else {
          //--- First stage of TR_BDF2
          if(TR_BDF2_stage == 1) {
            if(parameters.is_stationary == false)
              R_i += 1.0/parameters.time_step *
                     W[point][component_i] *
                     fe_v.shape_value_component(i, point, component_i) *
                     fe_v.JxW(point);

            for(unsigned int d = 0; d < dim; d++)
              R_i -= parameters.theta*flux[point][component_i][d] *
                     fe_v.shape_grad_component(i, point, component_i)[d] *
                     fe_v.JxW(point);

            R_i -= parameters.theta*forcing[point][component_i] *
                   fe_v.shape_value_component(i, point, component_i) *
                   fe_v.JxW(point);
          }
          //--- Second stage of TR_BDF2
          else if(TR_BDF2_stage == 2) {
            if(parameters.is_stationary == false)
              R_i += 1.0/parameters.time_step *
                     W[point][component_i] *
                     fe_v.shape_value_component(i, point, component_i) *
                     fe_v.JxW(point);

            for(unsigned int d = 0; d < dim; d++)
              R_i -= gamma_2*flux[point][component_i][d] *
                     fe_v.shape_grad_component(i, point, component_i)[d] *
                     fe_v.JxW(point);

            R_i -=  gamma_2*forcing[point][component_i] *
                    fe_v.shape_value_component(i, point, component_i) *
                    fe_v.JxW(point);
          }
        }

      }
      cell_rhs(i) -= R_i;
    }
  }


  // @sect4{ConservationLaw::assemble_rhs_face_term}
  //
  template<int dim>
  void ConservationLaw<dim>::assemble_rhs_face_term(const unsigned int       face_no,
                                                const FEFaceValuesBase<dim>& fe_v,
                                                const FEFaceValuesBase<dim>& fe_v_neighbor,
                                                const bool                   external_face,
                                                const unsigned int           boundary_id) {
    TimerOutput::Scope t(time_table, "Assemble rhs face term");

    cell_rhs = 0;

    const unsigned int n_q_points    = fe_v.n_quadrature_points;
    const unsigned int dofs_per_cell = fe_v.dofs_per_cell;

    Table<2, double> Wplus(n_q_points, EulerEquations<dim>::n_components),
                     Wminus(n_q_points, EulerEquations<dim>::n_components);

    for(unsigned int q = 0; q < n_q_points; ++q) {
      for(unsigned int i = 0; i < dofs_per_cell; ++i) {
        const unsigned int component_i = fe_v.get_fe().system_to_component_index(i).first;
        Wplus[q][component_i] += local_current_solution[i] *
                                 fe_v.shape_value_component(i, q, component_i);
      }
    }

    // Computing "opposite side" is a bit more complicated. If this is
    // an internal face, we can compute it as above by simply using the
    // independent variables from the neighbor:
    if(external_face == false) {
      for(unsigned int q = 0; q < n_q_points; ++q) {
        for(unsigned int i = 0; i < dofs_per_cell; ++i) {
          const unsigned int component_i = fe_v_neighbor.get_fe().system_to_component_index(i).first;
          Wminus[q][component_i] += local_current_solution_neighbor[i] *
                                    fe_v_neighbor.shape_value_component(i, q, component_i);
        }
      }
    }
    else {
      Assert(boundary_id < Parameters::AllParameters<dim>::max_n_boundaries,
             ExcIndexRange(boundary_id, 0, Parameters::AllParameters<dim>::max_n_boundaries));

      std::vector<Vector<double>> boundary_values(n_q_points, Vector<double>(EulerEquations<dim>::n_components));
      if(parameters.testcase == 0)
        parameters.boundary_conditions[boundary_id].values.vector_value_list(fe_v.get_quadrature_points(), boundary_values);
      else
        parameters.exact_solution.vector_value_list(fe_v.get_quadrature_points(), boundary_values);

      for(unsigned int q = 0; q < n_q_points; q++) {
        // Here we assume that boundary type, boundary normal vector and
        // boundary data values maintain the same during time advancing.
        EulerEquations<dim>::compute_Wminus(parameters.boundary_conditions[boundary_id].kind,
                                            fe_v.normal_vector(q), Wplus[q], boundary_values[q], Wminus[q]);
      }
    }

    std::vector<std::array<double, EulerEquations<dim>::n_components>> normal_fluxes(n_q_points);

    for(unsigned int q = 0; q < n_q_points; ++q)
      EulerEquations<dim>::numerical_normal_flux(fe_v.normal_vector(q), Wplus[q], Wminus[q], lambda_old, normal_fluxes[q]);

    for(unsigned int i = 0; i < dofs_per_cell; ++i) {
      if(fe_v.get_fe().has_support_on_face(i, face_no) == true) {
        double R_i = 0.0;

        for(unsigned int point = 0; point < n_q_points; ++point) {
          const unsigned int component_i = fe_v.get_fe().system_to_component_index(i).first;

          //--- Reorganizing contributions
          if(parameters.time_integration_scheme == "Theta_Method") {
            R_i += parameters.theta*normal_fluxes[point][component_i]*
                   fe_v.shape_value_component(i, point, component_i) *
                   fe_v.JxW(point);
          }
          //--- TR_BDF2 assembling
          else {
            //--- First stage of TR_BDF2
            if(TR_BDF2_stage == 1) {
              R_i += parameters.theta*normal_fluxes[point][component_i] *
                     fe_v.shape_value_component(i, point, component_i) *
                     fe_v.JxW(point);
            }
            //--- Second stage of TR_BDF2
            else if(TR_BDF2_stage == 2) {
              R_i += gamma_2*normal_fluxes[point][component_i]*
                     fe_v.shape_value_component(i, point, component_i) *
                     fe_v.JxW(point);
            }
          }
        }

        cell_rhs(i) -= R_i;
      }
    }
  }


  // @sect4{ConservationLaw::copy_local_to_global}
  //
  template<int dim>
  void ConservationLaw<dim>::copy_local_to_global(const std::vector<types::global_dof_index>& dof_indices) {
    TimerOutput::Scope t(time_table, "Copy local to global rhs");

    constraints.distribute_local_to_global(cell_rhs, dof_indices, right_hand_side);
    right_hand_side.compress(VectorOperation::add);
  }


  // @sect4{ConservationLaw::assemble_rhs}
  //
  template<int dim>
  void ConservationLaw<dim>::assemble_rhs() {
    TimerOutput::Scope t(time_table, "Assemble system");

    const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    const UpdateFlags update_flags = update_values | update_gradients |
                                     update_quadrature_points | update_JxW_values,
                      face_update_flags = update_values | update_quadrature_points |
                                          update_JxW_values |
                                          update_normal_vectors,
                      neighbor_face_update_flags = update_values;

    FEValues<dim>        fe_v(mapping, fe, quadrature, update_flags);
    FEFaceValues<dim>    fe_v_face(mapping, fe, face_quadrature, face_update_flags);
    FESubfaceValues<dim> fe_v_subface(mapping, fe, face_quadrature, face_update_flags);
    FEFaceValues<dim>    fe_v_face_neighbor(mapping, fe, face_quadrature, neighbor_face_update_flags);
    FESubfaceValues<dim> fe_v_subface_neighbor(mapping, fe, face_quadrature, neighbor_face_update_flags);

    // Then loop over all cells, initialize the FEValues object for the
    // current cell and call the function that assembles the problem on this
    // cell.
    for(const auto& cell: dof_handler.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        fe_v.reinit(cell);
        cell->get_dof_values(current_solution, local_current_solution);
        cell->get_dof_indices(dof_indices);

        assemble_rhs_cell_term(fe_v);
        copy_local_to_global(dof_indices);

        // Then loop over all the faces of this cell.  If a face is part of
        // the external boundary, then assemble boundary conditions there:
        for(unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no) {
          if(cell->at_boundary(face_no)) {
            fe_v_face.reinit(cell, face_no);
            assemble_rhs_face_term(face_no, fe_v_face, fe_v_face,
                                   true, cell->face(face_no)->boundary_id());
            copy_local_to_global(dof_indices);
          }

          // The alternative is that we are dealing with an internal face.
          else {
            if(cell->neighbor(face_no)->has_children()) {
              const unsigned int neighbor2 = cell->neighbor_of_neighbor(face_no);

              for(unsigned int subface_no = 0; subface_no < cell->face(face_no)->n_children(); ++subface_no) {
                const typename DoFHandler<dim>::active_cell_iterator
                neighbor_child = cell->neighbor_child_on_subface(face_no, subface_no);

                Assert(neighbor_child->face(neighbor2) == cell->face(face_no)->child(subface_no), ExcInternalError());
                Assert(neighbor_child->has_children() == false, ExcInternalError());

                fe_v_subface.reinit(cell, face_no, subface_no);
                fe_v_face_neighbor.reinit(neighbor_child, neighbor2);
                neighbor_child->get_dof_values(locally_relevant_solution, local_current_solution_neighbor);

                assemble_rhs_face_term(face_no, fe_v_subface, fe_v_face_neighbor,
                                       false, numbers::invalid_unsigned_int);
                copy_local_to_global(dof_indices);
              }
            }

            // The other possibility we have to care for is if the neighbor
            // is coarser than the current cell:
            else if(cell->neighbor(face_no)->level() != cell->level()) {
              const typename DoFHandler<dim>::cell_iterator neighbor = cell->neighbor(face_no);
              Assert(neighbor->level() == cell->level() - 1, ExcInternalError());

              const std::pair<unsigned int, unsigned int> faceno_subfaceno = cell->neighbor_of_coarser_neighbor(face_no);
              const unsigned int neighbor_face_no = faceno_subfaceno.first,
                                 neighbor_subface_no = faceno_subfaceno.second;

              Assert(neighbor->neighbor_child_on_subface(neighbor_face_no, neighbor_subface_no) == cell,
                     ExcInternalError());

              fe_v_face.reinit(cell, face_no);
              fe_v_subface_neighbor.reinit(neighbor, neighbor_face_no,
                                           neighbor_subface_no);
              neighbor->get_dof_values(locally_relevant_solution, local_current_solution_neighbor);

              assemble_rhs_face_term(face_no, fe_v_face, fe_v_subface_neighbor,
                                     false, numbers::invalid_unsigned_int);
              copy_local_to_global(dof_indices);
            }
            // Same refinement level
            else {
              fe_v_face.reinit(cell, face_no);
              const typename DoFHandler<dim>::cell_iterator neighbor = cell->neighbor(face_no);
              const unsigned int other_face_no = cell->neighbor_of_neighbor(face_no);
              fe_v_face_neighbor.reinit(neighbor, other_face_no);
              neighbor->get_dof_values(locally_relevant_solution, local_current_solution_neighbor);

              assemble_rhs_face_term(face_no, fe_v_face, fe_v_face_neighbor,
                                     false, numbers::invalid_unsigned_int);
              copy_local_to_global(dof_indices);
            }
          }
        }
      }
    }
  }


  // @sect4{ConservationLaw::solve}
  //
  // Here, we actually solve the linear system. The result of the computation will be
  // written into the argument vector passed to this function. The result is a
  // pair of number of iterations and the final linear residual.

  template<int dim>
  std::pair<unsigned int, double> ConservationLaw<dim>::solve(LinearAlgebra::distributed::Vector<double>& newton_update) {
    TimerOutput::Scope t(time_table, "Solve");

    switch(parameters.solver) {
      case Parameters::Solver::direct:
      {
        SolverControl  solver_control(1, 0);
        TrilinosWrappers::SolverDirect::AdditionalData data(parameters.output == Parameters::Solver::verbose);
        TrilinosWrappers::SolverDirect direct(solver_control, data);

        direct.solve(newton_update, right_hand_side_mf);

        return {solver_control.last_step(), solver_control.last_value()};
      }

      case Parameters::Solver::gmres:
      {
        SolverControl solver_control(parameters.max_iterations, parameters.linear_residual);
        SolverGMRES<LinearAlgebra::distributed::Vector<double>> gmres(solver_control);
        gmres.solve(matrix_free, newton_update, right_hand_side_mf, PreconditionIdentity());

        return {solver_control.last_step(), solver_control.last_value()};
      }
    }

    Assert(false, ExcNotImplemented());
    return {0, 0};
  }


  // @sect4{ConservationLaw::compute_refinement_indicators}

  // This function is real simple: We don't pretend that we know here what a
  // good refinement indicator would be. Rather, we assume that the
  // <code>EulerEquation</code> class would know about this, and so we simply
  // defer to the respective function we've implemented there:
  template<int dim>
  void ConservationLaw<dim>::compute_refinement_indicators(Vector<double>& refinement_indicators) const {
    EulerEquations<dim>::compute_refinement_indicators(dof_handler, mapping,
                                                       predictor, refinement_indicators);
  }


  // @sect4{ConservationLaw::refine_grid}

  // Here, we use the refinement indicators computed before and refine the
  // mesh. At the beginning, we loop over all cells and mark those that we
  // think should be refined:
  template<int dim>
  void ConservationLaw<dim>::refine_grid(const Vector<double>& refinement_indicators) {
    for(const auto& cell: dof_handler.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        const unsigned int cell_no = cell->active_cell_index();
        cell->clear_coarsen_flag();
        cell->clear_refine_flag();

        if(cell->level() < parameters.shock_levels &&
           std::fabs(refinement_indicators(cell_no)) > parameters.shock_val)
          cell->set_refine_flag();
        else if(cell->level() > 0 && std::fabs(refinement_indicators(cell_no)) < 0.75*parameters.shock_val)
          cell->set_coarsen_flag();
      }
    }
    MPI_Barrier(communicator);

    // Then we need to transfer the various solution vectors from the old to
    // the new grid while we do the refinement. The SolutionTransfer class is
    // our friend here; it has a fairly extensive documentation, including
    // examples, so we won't comment much on the following code. The last
    // three lines simply re-set the sizes of some other vectors to the now
    // correct size:

    // Transfer solution to vector that provides access to
    // locally relevant DoFs
    std::vector<LA::MPI::Vector> transfer_in(2);
    transfer_in[0].reinit(locally_owned_dofs, locally_relevant_dofs, communicator);
    transfer_in[1].reinit(locally_owned_dofs, locally_relevant_dofs, communicator);
    transfer_in[0] = old_solution;
    transfer_in[1] = predictor;

    // Initialize SolutionTransfer object and refine grid
    triangulation.prepare_coarsening_and_refinement();
    parallel::distributed::SolutionTransfer<dim,LA::MPI::Vector> soltrans(dof_handler);
    std::vector<const LA::MPI::Vector*> transfer_in_tmp(2);
    for(unsigned int index = 0; index < 2; ++index) {
      const LA::MPI::Vector* tmp = &transfer_in[index];
      transfer_in_tmp[index] = tmp;
    }
    const std::vector<const LA::MPI::Vector*> transfer_in_const = transfer_in_tmp;
    soltrans.prepare_for_coarsening_and_refinement(transfer_in_const);
    triangulation.execute_coarsening_and_refinement();

    // Distribute dofs and recreate locally_owned_dofs and locally_relevant_dofs index sets
    dof_handler.clear();
    dof_handler.distribute_dofs(fe);
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    // Create auxiliary vector to save the interpolation
    // (necessary because interpolate can be called only once
    // otherwise directly used old_solution and predictor)
    std::vector<LA::MPI::Vector*> transfer_out(2);
    transfer_out[0]->reinit(locally_owned_dofs, communicator);
    transfer_out[1]->reinit(locally_owned_dofs, communicator);

    // Interpolate
    soltrans.interpolate(transfer_out);

    // Assign the functions to the new interpolated
    old_solution.reinit(locally_owned_dofs, communicator);
    old_solution = *transfer_out[0];
    predictor.reinit(locally_owned_dofs, communicator);
    predictor = *transfer_out[1];
    current_solution.reinit(locally_owned_dofs, communicator);
    current_solution = old_solution;
    right_hand_side.reinit(locally_owned_dofs, communicator);
    right_hand_side_explicit.reinit(locally_owned_dofs, communicator);
  }


  // @sect4{ConservationLaw::compute_errors}

  // This function now is rather standard in computing errors. We use masks to extract
  // errors for single components.
  template<int dim>
  void ConservationLaw<dim>::compute_errors() {
    TimerOutput::Scope t(time_table, "Compute Errors");

    Vector<double> cellwise_errors(triangulation.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      locally_relevant_solution,
                                      parameters.exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm);
    const double error = VectorTools::compute_global_error(triangulation,
                                                           cellwise_errors,
                                                           VectorTools::L2_norm);
    const ComponentSelectFunction<dim> density_mask(EulerEquations<dim>::density_component,
                                                    EulerEquations<dim>::density_component + 1);
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      locally_relevant_solution,
                                      parameters.exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &density_mask);
    const double density_error = VectorTools::compute_global_error(triangulation,
                                                                   cellwise_errors,
                                                                   VectorTools::L2_norm);
    /*const ComponentSelectFunction<dim> velocity_mask(std::make_pair(EulerEquations<dim>::first_momentum_component,
                                                                    EulerEquations<dim>::first_momentum_component + dim),
                                                     EulerEquations<dim>::first_momentum_component + dim + 1);
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      locally_relevant_solution,
                                      parameters.exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &velocity_mask);
    const double velocity_error = VectorTools::compute_global_error(triangulation,
                                                                    cellwise_errors,
                                                                    VectorTools::L2_norm);*/
    const ComponentSelectFunction<dim> energy_mask(EulerEquations<dim>::energy_component,
                                                   EulerEquations<dim>::energy_component + 1);
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      locally_relevant_solution,
                                      parameters.exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &energy_mask);
    const double energy_error = VectorTools::compute_global_error(triangulation,
                                                                  cellwise_errors,
                                                                  VectorTools::L2_norm);
    if(Utilities::MPI::this_mpi_process(communicator) == 0) {
      output_error << error << std::endl;
      output_error << density_error << std::endl;
      //output_error << velocity_error << std::endl;
      output_error << energy_error << std::endl;
    }
  }


  // @sect4{ConservationLaw::output_results}

  // This function now is rather straightforward. All the magic, including
  // transforming data from conservative variables to physical ones has been
  // abstracted and moved into the EulerEquations class so that it can be
  // replaced in case we want to solve some other hyperbolic conservation law.
  //
  // Note that the number of the output file is determined by keeping a
  // counter in the form of a static variable that is set to zero the first
  // time we come to this function and is incremented by one at the end of
  // each invocation.
  template<int dim>
  void ConservationLaw<dim>::output_results() const {
    typename EulerEquations<dim>::Postprocessor postprocessor(parameters.schlieren_plot);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    data_out.add_data_vector(locally_relevant_solution,
                             EulerEquations<dim>::component_names(),
                             DataOut<dim>::type_dof_data,
                             EulerEquations<dim>::component_interpretation());

    data_out.add_data_vector(locally_relevant_solution, postprocessor);

    data_out.build_patches();

    static unsigned int output_file_number = 0;
    std::string         filename =
      "./" + parameters.dir + "/solution-" + Utilities::int_to_string(output_file_number, 3) + ".vtu";
    data_out.write_vtu_in_parallel(filename, communicator);

    ++output_file_number;
  }


  // @sect4{ConservationLaw::run}

  // This function contains the top-level logic of this program:
  // initialization, the time loop, and the inner Newton iteration.
  //
  // At the beginning, we read the mesh file specified by the parameter file,
  // setup the DoFHandler and various vectors, and then interpolate the given
  // initial conditions on this mesh. We then perform a number of mesh
  // refinements, based on the initial conditions, to obtain a mesh that is
  // already well adapted to the starting solution. At the end of this
  // process, we output the initial solution.
  template<int dim>
  void ConservationLaw<dim>::run() {
    make_grid_and_dofs();

    setup_system();

    if(parameters.testcase == 0)
      VectorTools::interpolate(dof_handler,
                               parameters.initial_conditions,
                               old_solution);
    else {
      parameters.exact_solution.set_time(0.0);
      VectorTools::interpolate(dof_handler,
                               parameters.exact_solution,
                               old_solution);
    }
    current_solution = old_solution;
    predictor        = old_solution;

    if(parameters.do_refine == true) {
      for(unsigned int i = 0; i < parameters.shock_levels; ++i) {
        Vector<double> refinement_indicators(triangulation.n_locally_owned_active_cells());

        compute_refinement_indicators(refinement_indicators);
        refine_grid(refinement_indicators);

        setup_system();

        VectorTools::interpolate(dof_handler,
                                 parameters.initial_conditions,
                                 old_solution);
        current_solution = old_solution;
        predictor        = old_solution;
      }
    }

    //TimerOutput::Scope t(time_table, "Output results");
    //output_results();

    // We then enter into the main time stepping loop. At the top we simply
    // output some status information so one can keep track of where a
    // computation is, as well as the header for a table that indicates
    // progress of the nonlinear inner iteration:
    LA::MPI::Vector newton_update;
    LinearAlgebra::distributed::Vector<double> newton_update_mf;
    LinearAlgebra::ReadWriteVector<double> newton_update_exchanger;
    newton_update.reinit(locally_owned_dofs, communicator);
    newton_update_mf.reinit(locally_owned_dofs, locally_relevant_dofs, communicator);
    newton_update_exchanger.reinit(locally_owned_dofs, communicator);

    double time = 0.0;
    double next_output = time + parameters.output_step;

    while(time < parameters.final_time) {
      pcout << "T=" << time << std::endl
            << "   Number of active cells:       "
            << triangulation.n_global_active_cells() << std::endl
            << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl
            << std::endl;

      pcout << "   NonLin Res     Lin Iter       Lin Res" << std::endl
            << "   _____________________________________" << std::endl;

      // Then comes the inner Newton iteration to solve the nonlinear
      // problem in each time step. The way it works is to reset matrix and
      // right hand side to zero, then assemble the linear system. If the
      // norm of the right hand side is small enough, then we declare that
      // the Newton iteration has converged. Otherwise, we solve the linear
      // system, update the current solution with the Newton increment, and
      // output convergence information. At the end, we check that the
      // number of Newton iterations is not beyond a limit of 10 -- if it
      // is, it appears likely that iterations are diverging and further
      // iterations would do no good. If that happens, we throw an exception
      // that will be caught in <code>main()</code> with status information
      // being displayed before the program aborts.
      //
      // Note that the way we write the AssertThrow macro below is by and
      // large equivalent to writing something like <code>if (!(nonlin_iter
      // @<= 10)) throw ExcMessage ("No convergence in nonlinear
      // solver");</code>. The only significant difference is that
      // AssertThrow also makes sure that the exception being thrown carries
      // with it information about the location (file name and line number)
      // where it was generated. This is not overly critical here, because
      // there is only a single place where this sort of exception can
      // happen; however, it is generally a very useful tool when one wants
      // to find out where an error occurred.
      unsigned int nonlin_iter = 0;
      unsigned int totlin_iter = 0;
      locally_relevant_old_solution = old_solution;
      //--- Restyling for TR_BDF2
      if(parameters.time_integration_scheme == "Theta_Method") {
        assemble_explicit_system();
        while(true) {
          if(parameters.testcase == 1)
            parameters.exact_solution.set_time(time + parameters.time_step);
          locally_relevant_solution = current_solution;
          right_hand_side = right_hand_side_explicit;
          assemble_rhs();
          right_hand_side_exchanger.import(right_hand_side, VectorOperation::insert);
          right_hand_side_mf.import(right_hand_side_exchanger, VectorOperation::insert);
          if(nonlin_iter == 0) {
            matrix_free.set_current_time(time + parameters.time_step);
            matrix_free.set_lambda_old(lambda_old);
            preconditioner.set_current_time(time + parameters.time_step);
            preconditioner.set_lambda_old(lambda_old);
          }

          const double res_norm = right_hand_side.l2_norm();
          if(std::fabs(res_norm) < 1e-10) {
            if(Utilities::MPI::this_mpi_process(communicator) == 0)
              std::printf("   %-16.3e (converged)\n\n", res_norm);
            break;
          }
          else {
            newton_update_mf = 0;

            std::pair<unsigned int, double> convergence = solve(newton_update_mf);
            newton_update_exchanger.import(newton_update_mf, VectorOperation::insert);
            newton_update.import(newton_update_exchanger, VectorOperation::insert);

            current_solution += newton_update;

            if(Utilities::MPI::this_mpi_process(communicator) == 0)
              std::printf("   %-16.3e %04d        %-5.2e\n",
                          res_norm, convergence.first, convergence.second);
            totlin_iter += convergence.first;
          }

          ++nonlin_iter;
          AssertThrow((res_norm < 1.0 || nonlin_iter < 20) && nonlin_iter <= 100,
                        ExcMessage("No convergence in nonlinear solver"));
        }
      }
      //--- TR_BDF2 Newton loops
      else {
        assemble_explicit_system();
        //--- First stage Newton loop
        while(true) {
          if(parameters.testcase == 1)
            parameters.exact_solution.set_time(time + 2.0*parameters.theta*parameters.time_step);
          locally_relevant_solution = current_solution;
          right_hand_side = right_hand_side_explicit;
          assemble_rhs();
          right_hand_side_exchanger.import(right_hand_side, VectorOperation::insert);
          right_hand_side_mf.import(right_hand_side_exchanger, VectorOperation::insert);
          if(nonlin_iter == 0) {
            matrix_free.set_current_time(time + 2.0*parameters.theta*parameters.time_step);
            matrix_free.set_lambda_old(lambda_old);
            matrix_free.set_TR_BDF2_stage(TR_BDF2_stage);
            preconditioner.set_current_time(time + 2.0*parameters.theta*parameters.time_step);
            preconditioner.set_lambda_old(lambda_old);
            preconditioner.set_TR_BDF2_stage(TR_BDF2_stage);
          }

          const double res_norm = right_hand_side.l2_norm();
          if(std::fabs(res_norm) < 1e-10) {
            if(Utilities::MPI::this_mpi_process(communicator) == 0)
              std::printf("   %-16.3e (converged)\n\n", res_norm);
            break;
          }
          else {
            newton_update_mf = 0;

            std::pair<unsigned int, double> convergence = solve(newton_update_mf);
            newton_update_exchanger.import(newton_update_mf, VectorOperation::insert);
            newton_update.import(newton_update_exchanger, VectorOperation::insert);

            current_solution += newton_update;

            if(Utilities::MPI::this_mpi_process(communicator) == 0)
              std::printf("   %-16.3e %04d        %-5.2e\n",
                          res_norm,
                          convergence.first,
                          convergence.second);
            totlin_iter += convergence.first;
          }

          ++nonlin_iter;
          AssertThrow((res_norm < 1.0 || nonlin_iter < 20) && nonlin_iter <= 100,
                        ExcMessage("No convergence in nonlinear solver"));
        }
        intermediate_solution = current_solution; //--- Save the solution of the first step
                                                  //---(fundamental for 1.0/Delta_t residual contribution)
        nonlin_iter = 0; //--- Reset iterations of Newton
        TR_BDF2_stage = 2; //--- Flag to specify that we have to pass at second stage
        assemble_explicit_system();
        //--- TR_BDF2 second stage Newton loop
        while(true) {
          if(parameters.testcase == 1)
            parameters.exact_solution.set_time(time + parameters.time_step);
          locally_relevant_solution = current_solution;
          right_hand_side = right_hand_side_explicit;
          assemble_rhs();
          right_hand_side_exchanger.import(right_hand_side, VectorOperation::insert);
          right_hand_side_mf.import(right_hand_side_exchanger, VectorOperation::insert);
          if(nonlin_iter == 0) {
            matrix_free.set_current_time(time + parameters.time_step);
            matrix_free.set_lambda_old(lambda_old);
            matrix_free.set_TR_BDF2_stage(TR_BDF2_stage);
            preconditioner.set_current_time(time + parameters.time_step);
            preconditioner.set_lambda_old(lambda_old);
            preconditioner.set_TR_BDF2_stage(TR_BDF2_stage);
          }

          const double res_norm = right_hand_side.l2_norm();
          if(std::fabs(res_norm) < 1e-10) {
            if(Utilities::MPI::this_mpi_process(communicator) == 0)
              std::printf("   %-16.3e (converged)\n\n", res_norm);
            break;
          }
          else {
            newton_update_mf = 0;

            std::pair<unsigned int, double> convergence = solve(newton_update_mf);
            newton_update_exchanger.import(newton_update_mf, VectorOperation::insert);
            newton_update.import(newton_update_exchanger, VectorOperation::insert);

            current_solution += newton_update;

            if(Utilities::MPI::this_mpi_process(communicator) == 0)
              std::printf("   %-16.3e %04d        %-5.2e\n",
                          res_norm,
                          convergence.first,
                          convergence.second);
            totlin_iter += convergence.first;
          }

          ++nonlin_iter;
          AssertThrow((res_norm < 1.0 || nonlin_iter < 20) && nonlin_iter <= 100,
                       ExcMessage("No convergence in nonlinear solver"));
        }
        TR_BDF2_stage = 1; //--- Flag to specify that we have to pass at first stage in the next step
      }
      if(Utilities::MPI::this_mpi_process(communicator) == 0)
        output_niter << totlin_iter <<std::endl;

      // We only get to this point if the Newton iteration has converged, so
      // do various post convergence tasks here:
      //
      // First, we update the time and produce graphical output if so
      // desired. Then we update a predictor for the solution at the next
      // time step by approximating $\mathbf w^{n+1}\approx \mathbf w^n +
      // \delta t \frac{\partial \mathbf w}{\partial t} \approx \mathbf w^n
      // + \delta t \; \frac{\mathbf w^n-\mathbf w^{n-1}}{\delta t} = 2
      // \mathbf w^n - \mathbf w^{n-1}$ to try and make adaptivity work
      // better.  The idea is to try and refine ahead of a front, rather
      // than stepping into a coarse set of elements and smearing the
      // old_solution.  This simple time extrapolator does the job. With
      // this, we then refine the mesh if so desired by the user, and
      // finally continue on with the next time step:
      //if(parameters.testcase == 1)
      //  compute_errors();

      time += parameters.time_step;

      if(parameters.output_step < 0)
        output_results();
      else if(time >= next_output) {
        TimerOutput::Scope t(time_table, "Output results");
        output_results();
        next_output += parameters.output_step;
      }

      predictor = current_solution;
      predictor.sadd(2.0, -1.0, old_solution);

      old_solution = current_solution;

      if(parameters.do_refine == true) {
        Vector<double> refinement_indicators(triangulation.n_active_cells());
        compute_refinement_indicators(refinement_indicators);

        refine_grid(refinement_indicators);
        setup_system();

        newton_update.reinit(locally_relevant_dofs, communicator);
      }
      time_table.print_wall_time_statistics(communicator);
    }
  }
} // namespace Step33

// @sect3{main()}

// The following ``main'' function is similar to previous examples and need
// not to be commented on. Note that the program aborts if no input file name
// is given on the command line.
int main(int argc, char *argv[]) {
  try {
    using namespace dealii;
    using namespace Step33;

    if(argc != 2) {
      std::cout << "Usage:" << argv[0] << " input_file" << std::endl;
      std::exit(1);
    }

    ParameterHandler prm;
    Parameters::AllParameters<2>::declare_parameters(prm);
    prm.parse_input(argv[1]);

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, dealii::numbers::invalid_unsigned_int);

    ConservationLaw<2> cons(prm);
    cons.run();
  }
  catch (std::exception &exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch (...)  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  };

  return 0;
}

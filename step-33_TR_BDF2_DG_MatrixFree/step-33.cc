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

// Extra include (in particular for MatrixFree)
#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/lac/solver_gmres.h>

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
    static const unsigned int fe_degree                = 2;
    static const unsigned int n_q_points_1d            = fe_degree + 1;

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


    static std::vector<DataComponentInterpretation::DataComponentInterpretation> component_interpretation() {
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
    // the pressure and the speed of sound from a vector of conserved variables. This we can do based on
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

    // We define the flux function $F(W)$ as one large matrix. Each row of
    // this matrix represents a scalar conservation law for the component in
    // that row. Note that we know the size of the matrix: it has as many
    // rows as the system has components, and <code>dim</code> columns; rather
    // than using a FullMatrix object for such a matrix (which has a variable
    // number of rows and columns and must therefore allocate memory on the
    // heap each time such a matrix is created), we use a rectangular array of
    // numbers right away.
    //
    // We templatize the numerical type of the flux function so that we may
    // use the automatic differentiation type here. Similarly, we will call
    // the function with different input vector data types, so we templatize
    // on it as well:
    template<typename InputVector>
    static void compute_flux_matrix(const InputVector& W,
                                    Tensor<1, n_components, Tensor<1, dim, typename InputVector::value_type>>& flux) {
      // First compute the pressure that appears in the flux matrix, and then
      // compute the first <code>dim</code> columns of the matrix that
      // correspond to the momentum terms:
      const auto pressure = compute_pressure(W);

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
    // numerical flux function to enforce boundary conditions. This routine
    // is the basic Lax-Friedrich's flux. It's form has also been given already in the introduction:
    template<typename InputVector, typename NormalVector, typename value_type>
    static void numerical_normal_flux(const NormalVector& normal,
                                      const InputVector&  Wplus,
                                      const InputVector&  Wminus,
                                      const value_type    gamma,
                                      Tensor<1, n_components, typename InputVector::value_type>& normal_flux) {
      Tensor<1, n_components, Tensor<1, dim, typename InputVector::value_type>>  iflux, oflux;

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
                                       Tensor<1, n_components, typename InputVector::value_type>& forcing,
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
        std::fill(forcing.begin_raw(), forcing.end_raw(), 0.0);
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
    template<typename DataVector, typename NormalVector>
    static void compute_Wminus(const std::array<BoundaryKind, n_components>& boundary_kind,
                               const NormalVector&                           normal_vector,
                               const DataVector&                             Wplus,
                               const Vector<double>&                         boundary_values,
                               const DataVector&                             Wminus) {
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
             kinetic_energy *= 0.5/density;

             Wminus[c] = boundary_values(c)/(gas_gamma - 1.0) + kinetic_energy;

             break;
          }

          case no_penetration_boundary:
          {
            // We prescribe the velocity (we are dealing with a particular
            // component here so that the average of the velocities is
            // orthogonal to the surface normal. This creates sensitivities
            // of across the velocity components.
            typename DataVector::value_type v_dot_n = 0;
            for(unsigned int d = 0; d < dim; d++)
              v_dot_n += Wplus[d]*normal_vector[d];

            Wminus[c] = Wplus[c] - 2.0*v_dot_n*normal_vector[c];
            break;
          }

          default:
            Assert(false, ExcNotImplemented());
        }
      }
    }


    template<typename DataVector, typename NormalVector>
    static void compute_Wminus(const std::array<BoundaryKind, n_components>& boundary_kind,
                               const NormalVector&                           normal_vector,
                               const DataVector&                             Wplus,
                               const Vector<double>&                         boundary_values,
                               DataVector&                                   Wminus) {
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
             kinetic_energy *= 0.5/density;

             Wminus[c] = boundary_values(c)/(gas_gamma - 1.0) + kinetic_energy;

             break;
          }

          case no_penetration_boundary:
          {
            // We prescribe the velocity (we are dealing with a particular
            // component here so that the average of the velocities is
            // orthogonal to the surface normal. This creates sensitivities
            // of across the velocity components.
            typename DataVector::value_type v_dot_n = 0;
            for(unsigned int d = 0; d < dim; d++)
              v_dot_n += Wplus[d]*normal_vector[d];

            Wminus[c] = Wplus[c] - 2.0*v_dot_n*normal_vector[c];
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
    template<typename InputVector>
    static void compute_refinement_indicators(const DoFHandler<dim>& dof_handler,
                                              const Mapping<dim>&    mapping,
                                              const InputVector&     solution,
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
    // interfaces.
    //
    class Postprocessor : public DataPostprocessor<dim> {
    public:
      Postprocessor(const bool do_schlieren_plot);

      virtual void evaluate_vector_field(const DataPostprocessorInputs::Vector<dim>& inputs,
                                         std::vector<Vector<double>>&                computed_quantities) const override;

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
                                                                 std::vector<Vector<double>>&  computed_quantities) const {
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
      Assert(inputs.solution_gradients.size() == n_quadrature_points, ExcInternalError());

    Assert(computed_quantities.size() == n_quadrature_points, ExcInternalError());

    Assert(inputs.solution_values[0].size() == n_components, ExcInternalError());

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
      int    global_refinements; //--- Number of initial refinements

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
    struct AllParameters: public Solver,
                          public Refinement,
                          public Output {
      static const unsigned int max_n_boundaries = 10;

      struct BoundaryConditions {
        std::array<typename EulerEquations<dim>::BoundaryKind, EulerEquations<dim>::n_components>  kind;

        FunctionParser<dim> values;

        BoundaryConditions();
      };

      //--- Auxiliary class in case exact solution is available
      struct ExactSolution: public Function<dim> {
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
      std::fill(kind.begin(), kind.end(), EulerEquations<dim>::no_penetration_boundary);
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
    AllParameters<dim>::AllParameters(): time_step(1.0),
                                         final_time(1.0),
                                         theta(0.5),
                                         is_stationary(true),
                                         testcase(0),
                                         initial_conditions(EulerEquations<dim>::n_components),
                                         exact_solution(ExactSolution(0.0)) {}

    // Struct AllParameters constructor
    template<int dim>
    AllParameters<dim>::AllParameters(ParameterHandler& prm): time_step(1.0),
                                                              final_time(1.0),
                                                              theta(0.5),
                                                              is_stationary(true),
                                                              testcase(0),
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

  template<int dim, int fe_degree, int n_q_points_1d, typename Number>
  class ConservationLawOperator: public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<Number>> {
  public:
    ConservationLawOperator();

    ConservationLawOperator(ParameterHandler& prm);

    void set_TR_BDF2_stage(const unsigned int stage);

    double get_lambda_old() const;

    void vmult_explicit(const double                                                   current_time,
                        const std::vector<LinearAlgebra::distributed::Vector<Number>>& src,
                        LinearAlgebra::distributed::Vector<Number>&                    dst);

    void vmult_rhs(const double                                      next_time,
                   const LinearAlgebra::distributed::Vector<Number>& src,
                   LinearAlgebra::distributed::Vector<Number>&       dst);

    void vmult(LinearAlgebra::distributed::Vector<Number>&       dst,
               const LinearAlgebra::distributed::Vector<Number>& src) const;

    virtual void compute_diagonal() override {}

  protected:
    Parameters::AllParameters<dim> parameters;  //--- Auxiliary variable to read parameters

    double       lambda_old;     //--- Extra parameter to save old Lax-Friedrichs stability (for explicit caching)
    double       gamma_2;        //--- Extra parameter of TR_BDF2
    double       gamma_3;        //--- Extra parameter of TR_BDF2
    unsigned int TR_BDF2_stage;  //--- Extra parameter for counting stage TR_BDF2

    virtual void apply_add(LinearAlgebra::distributed::Vector<Number>&       dst,
                           const LinearAlgebra::distributed::Vector<Number>& src) const override {}

  private:
    void assemble_explicit_cell_term(const MatrixFree<dim, Number>&                                 data,
                                     LinearAlgebra::distributed::Vector<Number>&                    dst,
                                     const std::vector<LinearAlgebra::distributed::Vector<Number>>& src,
                                     const std::pair<unsigned int, unsigned int>&                   cell_range);
    void assemble_explicit_face_term(const MatrixFree<dim, Number>&                                 data,
                                     LinearAlgebra::distributed::Vector<Number>&                    dst,
                                     const std::vector<LinearAlgebra::distributed::Vector<Number>>& src,
                                     const std::pair<unsigned int, unsigned int>&                   face_range);
    void assemble_explicit_boundary_term(const MatrixFree<dim, Number>&                                 data,
                                         LinearAlgebra::distributed::Vector<Number>&                    dst,
                                         const std::vector<LinearAlgebra::distributed::Vector<Number>>& src,
                                         const std::pair<unsigned int, unsigned int>&                   face_range);

    void assemble_rhs_cell_term(const MatrixFree<dim, Number>&                    data,
                                LinearAlgebra::distributed::Vector<Number>&       dst,
                                const LinearAlgebra::distributed::Vector<Number>& src,
                                const std::pair<unsigned int, unsigned int>& cell_range);
    void assemble_rhs_face_term(const MatrixFree<dim, Number>&                    data,
                                LinearAlgebra::distributed::Vector<Number>&       dst,
                                const LinearAlgebra::distributed::Vector<Number>& src,
                                const std::pair<unsigned int, unsigned int>&      face_range);
    void assemble_rhs_boundary_term(const MatrixFree<dim, Number>&                    data,
                                    LinearAlgebra::distributed::Vector<Number>&       dst,
                                    const LinearAlgebra::distributed::Vector<Number>& src,
                                    const std::pair<unsigned int, unsigned int>&      face_range);

    void assemble_cell_term(const MatrixFree<dim, Number>&                    data,
                            LinearAlgebra::distributed::Vector<Number>&       dst,
                            const LinearAlgebra::distributed::Vector<Number>& src,
                            const std::pair<unsigned int, unsigned int>& cell_range);
    void assemble_face_term(const MatrixFree<dim, Number>&                    data,
                            LinearAlgebra::distributed::Vector<Number>&       dst,
                            const LinearAlgebra::distributed::Vector<Number>& src,
                            const std::pair<unsigned int, unsigned int>&      face_range);
    void assemble_boundary_term(const MatrixFree<dim, Number>&                    data,
                                LinearAlgebra::distributed::Vector<Number>&       dst,
                                const LinearAlgebra::distributed::Vector<Number>& src,
                                const std::pair<unsigned int, unsigned int>&      face_range);
  };

  // Default constructor
  template<int dim, int fe_degree, int n_q_points_1d, typename Number>
  ConservationLawOperator<dim, fe_degree, n_q_points_1d, Number>::ConservationLawOperator():
    MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<Number>>() {}

  // Constructor with parameters (I need time step and parameter of Theta_Method or TR-BDF2 scheme)
  template<int dim, int fe_degree, int n_q_points_1d, typename Number>
  ConservationLawOperator<dim, fe_degree, n_q_points_1d, Number>::ConservationLawOperator(ParameterHandler& prm):
    MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<Number>>(), parameters(prm) {
      if(parameters.time_integration_scheme == "TR_BDF2") {
        parameters.theta = 1.0 - std::sqrt(2)/2.0;
        gamma_2 = (1.0 - 2.0*parameters.theta)/(2.0*(1.0 - parameters.theta));
        gamma_3 = (1.0 - gamma_2)/(2.0*parameters.theta);
        TR_BDF2_stage = 1;
      }
  }

  // Setter of TR-BDF2 stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  template<int dim, int fe_degree, int n_q_points_1d, typename Number>
  void ConservationLawOperator<dim, fe_degree, n_q_points_1d, Number>::set_TR_BDF2_stage(const unsigned int stage) {
    Assert(stage == 1 || stage == 2, ExcInternalError());
    TR_BDF2_stage = stage;
  }

  // Getter for lambda_old (the only frozen contribution in the Jacobian)
  template<int dim, int fe_degree, int n_q_points_1d, typename Number>
  double ConservationLawOperator<dim, fe_degree, n_q_points_1d, Number>::get_lambda_old() const {
    return lambda_old;
  }

  // Assemble explicit contribution of cells
  template<int dim, int fe_degree, int n_q_points_1d, typename Number>
  void ConservationLawOperator<dim, fe_degree, n_q_points_1d, Number>::
  assemble_explicit_cell_term(const MatrixFree<dim, Number>&                                 data,
                              LinearAlgebra::distributed::Vector<Number>&                    dst,
                              const std::vector<LinearAlgebra::distributed::Vector<Number>>& src,
                              const std::pair<unsigned int, unsigned int>&                   cell_range) {
    Tensor<1, EulerEquations<dim>::n_components, Tensor<1, dim, VectorizedArray<Number>>> flux_old;
    Tensor<1, EulerEquations<dim>::n_components, VectorizedArray<Number>>                 forcing_old;

    FEEvaluation<dim, fe_degree, n_q_points_1d, EulerEquations<dim>::n_components, Number> phi_old(data), phi_int(data);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_old.reinit(cell);
      phi_int.reinit(cell);
      phi_old.gather_evaluate(src[0], true, false);
      phi_int.gather_evaluate(src[1], true, false);
      for(unsigned int q = 0; q < phi_old.n_q_points; ++q) {
        const auto W_old = phi_old.get_value(q);
        EulerEquations<dim>::compute_flux_matrix(W_old, flux_old);
        EulerEquations<dim>::compute_forcing_vector(W_old, forcing_old, parameters.testcase);
        if(parameters.time_integration_scheme == "Theta_Method") {
          phi_old.submit_gradient((1.0 - parameters.theta)*flux_old, q);
          if(parameters.is_stationary == false)
            phi_old.submit_value(1.0/parameters.time_step*W_old + (1.0 - parameters.theta)*forcing_old, q);
          else
            phi_old.submit_value((1.0 - parameters.theta)*forcing_old, q);
        }
        //--- Stages of TR-BDF2
        else {
          //--- First stage of TR_BDF2
          if(TR_BDF2_stage == 1) {
            phi_old.submit_gradient(parameters.theta*flux_old, q);
            if(parameters.is_stationary == false)
              phi_old.submit_value(1.0/parameters.time_step*W_old + parameters.theta*forcing_old, q);
            else
              phi_old.submit_value(parameters.theta*forcing_old, q);
          }
          //--- Second stage of TR_BDF2
          else if(TR_BDF2_stage == 2) {
            const auto W_int = phi_int.get_value(q);
            if(parameters.is_stationary == false)
              phi_old.submit_value(1.0/parameters.time_step*(gamma_3*W_int + (1.0 - gamma_3)*W_old), q);
          }
        }
      }
      if(TR_BDF2_stage == 2)
        phi_old.integrate_scatter(true, false, dst);
      else
        phi_old.integrate_scatter(true, true, dst);
    }
  }

  // Assemble explicit contribution for interior faces
  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void ConservationLawOperator<dim, fe_degree, n_q_points_1d, Number>::
  assemble_explicit_face_term(const MatrixFree<dim, Number>&                                 data,
                              LinearAlgebra::distributed::Vector<Number>&                    dst,
                              const std::vector<LinearAlgebra::distributed::Vector<Number>>& src,
                              const std::pair<unsigned int, unsigned int>&                   face_range) {
    Tensor<1, EulerEquations<dim>::n_components, VectorizedArray<Number>> normal_fluxes_old;

    FEFaceEvaluation<dim, fe_degree, n_q_points_1d, EulerEquations<dim>::n_components, Number> phi_m(data, false);
    FEFaceEvaluation<dim, fe_degree, n_q_points_1d, EulerEquations<dim>::n_components, Number> phi_p(data, true);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_p.reinit(face);
      phi_p.gather_evaluate(src[0], true, false);

      phi_m.reinit(face);
      phi_m.gather_evaluate(src[0], true, false);

      for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
        const auto Wminus_old = phi_m.get_value(q);
        const auto Wplus_old  = phi_p.get_value(q);
        lambda_old = std::max(EulerEquations<dim>::compute_velocity(Wplus_old)  +
                              EulerEquations<dim>::compute_speed_of_sound(Wplus_old),
                              EulerEquations<dim>::compute_velocity(Wminus_old) +
                              EulerEquations<dim>::compute_speed_of_sound(Wminus_old))[0];
        EulerEquations<dim>::numerical_normal_flux(phi_p.get_normal_vector(q), Wplus_old, Wminus_old, lambda_old, normal_fluxes_old);
        if(parameters.time_integration_scheme == "Theta_Method") {
          phi_m.submit_value((1.0 - parameters.theta)*normal_fluxes_old, q);
          phi_p.submit_value((parameters.theta - 1.0)*normal_fluxes_old, q);
        }
        else {
          if(TR_BDF2_stage == 1) {
            phi_m.submit_value(parameters.theta*normal_fluxes_old, q);
            phi_p.submit_value(-parameters.theta*normal_fluxes_old, q);
          }
        }
      }
      if(TR_BDF2_stage == 1) {
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
  }

  // Assemble explicit contribution for boundary faces
  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void ConservationLawOperator<dim, fe_degree, n_q_points_1d, Number>::
  assemble_explicit_boundary_term(const MatrixFree<dim, Number>&                                 data,
                                  LinearAlgebra::distributed::Vector<Number>&                    dst,
                                  const std::vector<LinearAlgebra::distributed::Vector<Number>>& src,
                                  const std::pair<unsigned int, unsigned int>&                   face_range) {
    Tensor<1, EulerEquations<dim>::n_components, VectorizedArray<Number>> normal_fluxes_old;
    Tensor<1, EulerEquations<dim>::n_components, VectorizedArray<Number>> Wminus_old;

    FEFaceEvaluation<dim, fe_degree, n_q_points_1d, EulerEquations<dim>::n_components, Number> phi(data, true);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi.reinit(face);
      phi.gather_evaluate(src[0], true, false);

      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto Wplus_old = phi.get_value(q);
        const auto boundary_id = data.get_boundary_id(face);
        Assert(boundary_id < Parameters::AllParameters<dim>::max_n_boundaries,
               ExcIndexRange(boundary_id, 0, Parameters::AllParameters<dim>::max_n_boundaries));
        Vector<double> boundary_values(EulerEquations<dim>::n_components);
        const auto& p_vectorized = phi.quadrature_point(q);
        Point<dim> p;
        for(unsigned d = 0; d < dim; ++d)
          p[d] = p_vectorized[d][0];
        if(parameters.testcase == 0)
          parameters.boundary_conditions[boundary_id].values.vector_value(p, boundary_values);
        else
          parameters.exact_solution.vector_value(p, boundary_values);
        EulerEquations<dim>::compute_Wminus(parameters.boundary_conditions[boundary_id].kind,
                                            phi.get_normal_vector(q), Wplus_old, boundary_values, Wminus_old);
        lambda_old = std::max(EulerEquations<dim>::compute_velocity(Wplus_old)  +
                              EulerEquations<dim>::compute_speed_of_sound(Wplus_old),
                              EulerEquations<dim>::compute_velocity(Wminus_old) +
                              EulerEquations<dim>::compute_speed_of_sound(Wminus_old))[0];
        EulerEquations<dim>::numerical_normal_flux(phi.get_normal_vector(q), Wplus_old, Wminus_old, lambda_old, normal_fluxes_old);
        if(parameters.time_integration_scheme == "Theta_Method")
          phi.submit_value((parameters.theta - 1.0)*normal_fluxes_old, q);
        else {
          if(TR_BDF2_stage == 1)
            phi.submit_value(-parameters.theta*normal_fluxes_old, q);
        }
      }
      if(TR_BDF2_stage == 1)
        phi.integrate_scatter(true, false, dst);
    }
  }

  // Collect all the contributions and execute loop
  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void ConservationLawOperator<dim, fe_degree, n_q_points_1d, Number>::
  vmult_explicit(const double                                                   current_time,
                 const std::vector<LinearAlgebra::distributed::Vector<Number>>& src,
                 LinearAlgebra::distributed::Vector<Number>&                    dst) {
    if(parameters.testcase == 1)
      parameters.exact_solution.set_time(current_time);

    this->data->loop(&ConservationLawOperator::assemble_explicit_cell_term,
                     &ConservationLawOperator::assemble_explicit_face_term,
                     &ConservationLawOperator::assemble_explicit_boundary_term,
                     this,
                     dst,
                     src,
                     true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  // Assemble rhs contribution of cells
  template<int dim, int fe_degree, int n_q_points_1d, typename Number>
  void ConservationLawOperator<dim, fe_degree, n_q_points_1d, Number>::
  assemble_rhs_cell_term(const MatrixFree<dim, Number>&                    data,
                         LinearAlgebra::distributed::Vector<Number>&       dst,
                         const LinearAlgebra::distributed::Vector<Number>& src,
                         const std::pair<unsigned int, unsigned int>&      cell_range) {
    Tensor<1, EulerEquations<dim>::n_components, Tensor<1, dim, VectorizedArray<Number>>> flux;
    Tensor<1, EulerEquations<dim>::n_components, VectorizedArray<Number>>                 forcing;

    FEEvaluation<dim, fe_degree, n_q_points_1d, EulerEquations<dim>::n_components, Number> phi(data);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, true, false);
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto W = phi.get_value(q);
        EulerEquations<dim>::compute_flux_matrix(W, flux);
        EulerEquations<dim>::compute_forcing_vector(W, forcing, parameters.testcase);
        if(parameters.time_integration_scheme == "Theta_Method") {
          phi.submit_gradient(parameters.theta*flux, q);
          if(parameters.is_stationary == false)
            phi.submit_value(-1.0/parameters.time_step*W + parameters.theta*forcing, q);
          else
            phi.submit_value(parameters.theta*forcing, q);
        }
        //--- Stages of TR-BDF2
        else {
          //--- First stage of TR_BDF2
          if(TR_BDF2_stage == 1) {
            phi.submit_gradient(parameters.theta*flux, q);
            if(parameters.is_stationary == false)
              phi.submit_value(-1.0/parameters.time_step*W + parameters.theta*forcing, q);
            else
              phi.submit_value(parameters.theta*forcing, q);
          }
          //--- Second stage of TR_BDF2
          else if(TR_BDF2_stage == 2) {
            phi.submit_gradient(gamma_2*flux, q);
            if(parameters.is_stationary == false)
              phi.submit_value(-1.0/parameters.time_step*W + gamma_2*forcing, q);
            else
              phi.submit_value(gamma_2*forcing, q);
          }
        }
      }
      phi.integrate_scatter(true, true, dst);
    }
  }

  // Assemble right_hand_side contribution for interior faces
  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void ConservationLawOperator<dim, fe_degree, n_q_points_1d, Number>::
  assemble_rhs_face_term(const MatrixFree<dim, Number>&                    data,
                         LinearAlgebra::distributed::Vector<Number>&       dst,
                         const LinearAlgebra::distributed::Vector<Number>& src,
                         const std::pair<unsigned int, unsigned int>&      face_range) {
    Tensor<1, EulerEquations<dim>::n_components, VectorizedArray<Number>> normal_fluxes;

    FEFaceEvaluation<dim, fe_degree, n_q_points_1d, EulerEquations<dim>::n_components, Number> phi_m(data, false);
    FEFaceEvaluation<dim, fe_degree, n_q_points_1d, EulerEquations<dim>::n_components, Number> phi_p(data, true);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_p.reinit(face);
      phi_p.gather_evaluate(src, true, false);

      phi_m.reinit(face);
      phi_m.gather_evaluate(src, true, false);

      for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
        const auto Wminus = phi_m.get_value(q);
        const auto Wplus  = phi_p.get_value(q);
        EulerEquations<dim>::numerical_normal_flux(phi_p.get_normal_vector(q), Wplus, Wminus, lambda_old, normal_fluxes);
        if(parameters.time_integration_scheme == "Theta_Method") {
          phi_m.submit_value(parameters.theta*normal_fluxes, q);
          phi_p.submit_value(-parameters.theta*normal_fluxes, q);
        }
        else {
          if(TR_BDF2_stage == 1) {
            phi_m.submit_value(parameters.theta*normal_fluxes, q);
            phi_p.submit_value(-parameters.theta*normal_fluxes, q);
          }
          else {
            phi_m.submit_value(gamma_2*normal_fluxes, q);
            phi_p.submit_value(-gamma_2*normal_fluxes, q);
          }
        }
      }
      phi_p.integrate_scatter(true, false, dst);
      phi_m.integrate_scatter(true, false, dst);
    }
  }

  // Assemble explicit contribution for boundary faces
  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void ConservationLawOperator<dim, fe_degree, n_q_points_1d, Number>::
  assemble_rhs_boundary_term(const MatrixFree<dim, Number>&                    data,
                             LinearAlgebra::distributed::Vector<Number>&       dst,
                             const LinearAlgebra::distributed::Vector<Number>& src,
                             const std::pair<unsigned int, unsigned int>&      face_range) {
    Tensor<1, EulerEquations<dim>::n_components, VectorizedArray<Number>> normal_fluxes;
    Tensor<1, EulerEquations<dim>::n_components, VectorizedArray<Number>> Wminus;

    FEFaceEvaluation<dim, fe_degree, n_q_points_1d, EulerEquations<dim>::n_components, Number> phi(data, true);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi.reinit(face);
      phi.gather_evaluate(src, true, false);

      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto Wplus = phi.get_value(q);
        const auto boundary_id = data.get_boundary_id(face);
        Assert(boundary_id < Parameters::AllParameters<dim>::max_n_boundaries,
               ExcIndexRange(boundary_id, 0, Parameters::AllParameters<dim>::max_n_boundaries));
        Vector<double> boundary_values(EulerEquations<dim>::n_components);
        const auto& p_vectorized = phi.quadrature_point(q);
        Point<dim> p;
        for(unsigned d = 0; d < dim; ++d)
          p[d] = p_vectorized[d][0];
        if(parameters.testcase == 0)
          parameters.boundary_conditions[boundary_id].values.vector_value(p, boundary_values);
        else
          parameters.exact_solution.vector_value(p, boundary_values);
        EulerEquations<dim>::compute_Wminus(parameters.boundary_conditions[boundary_id].kind,
                                            phi.get_normal_vector(q), Wplus, boundary_values, Wminus);
        EulerEquations<dim>::numerical_normal_flux(phi.get_normal_vector(q), Wplus, Wminus, lambda_old, normal_fluxes);
        if(parameters.time_integration_scheme == "Theta_Method")
          phi.submit_value(-parameters.theta*normal_fluxes, q);
        else {
          if(TR_BDF2_stage == 1)
            phi.submit_value(-parameters.theta*normal_fluxes, q);
          else
            phi.submit_value(-gamma_2*normal_fluxes, q);
        }
      }
      phi.integrate_scatter(true, false, dst);
    }
  }

  // Collect all the contributions and execute loop
  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void ConservationLawOperator<dim, fe_degree, n_q_points_1d, Number>::
  vmult_rhs(const double                                      next_time,
            const LinearAlgebra::distributed::Vector<Number>& src,
            LinearAlgebra::distributed::Vector<Number>&       dst) {
    if(parameters.testcase == 1)
      parameters.exact_solution.set_time(next_time);

    this->data->loop(&ConservationLawOperator::assemble_rhs_cell_term,
                     &ConservationLawOperator::assemble_rhs_face_term,
                     &ConservationLawOperator::assemble_rhs_boundary_term,
                     this,
                     dst,
                     src,
                     true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  // Assemble matrix-vector action for cells
  template<int dim, int fe_degree, int n_q_points_1d, typename Number>
  void ConservationLawOperator<dim, fe_degree, n_q_points_1d, Number>::
  assemble_cell_term(const MatrixFree<dim, Number>&                    data,
                     LinearAlgebra::distributed::Vector<Number>&       dst,
                     const LinearAlgebra::distributed::Vector<Number>& src,
                     const std::pair<unsigned int, unsigned int>&      cell_range) {
    Tensor<1, EulerEquations<dim>::n_components, Tensor<1, dim, VectorizedArray<Sacado::Fad::DFad<Number>>>> flux;
    Tensor<1, EulerEquations<dim>::n_components, VectorizedArray<Sacado::Fad::DFad<Number>>>                 forcing;

    const unsigned int dofs_per_cell      = data.get_dofs_per_cell();
    const unsigned int dofs_per_component = dofs_per_cell/EulerEquations<dim>::n_components;

    std::vector<unsigned int> local_dof_indices(dofs_per_cell);
    FullMatrix<double>        cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>            cell_src(dofs_per_cell), cell_dst(dofs_per_cell);

    //std::vector<Sacado::Fad::DFad<Number>> independent_local_dof_values(dofs_per_cell);

    FEEvaluation<dim, fe_degree, n_q_points_1d, EulerEquations<dim>::n_components, Number> phi(data);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.read_dof_values(src);
      for(unsigned int i = 0; i < dofs_per_component; ++i) {
        const auto& tmp_dof_getter = phi.get_dof_value(i);
        decltype(tmp_dof_getter) tmp_dof_filler(EulerEquations<dim>::n_components);
        for(unsigned int j = 0; j < EulerEquations<dim>::n_components; ++j) {
          independent_local_dof_values[EulerEquations<dim>::n_components*i + j] = tmp_dof_getter[j];
          independent_local_dof_values[EulerEquations<dim>::n_components*i + j].diff(EulerEquations<dim>::n_components*i + j,
                                                                                     dofs_per_cell);
          tmp_dof_filler[j] = independent_local_dof_values[EulerEquations<dim>::n_components*i + j];
        }
        phi.submit_dof_value(tmp_dof_filler);
      }
      phi.evaluate(true, false);
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto W = phi.get_value(q);
        EulerEquations<dim>::compute_flux_matrix(W, flux);
        EulerEquations<dim>::compute_forcing_vector(W, forcing, parameters.testcase);
        if(parameters.time_integration_scheme == "Theta_Method") {
          phi.submit_gradient(parameters.theta*flux, q);
          if(parameters.is_stationary == false)
            phi.submit_value(-1.0/parameters.time_step*W + parameters.theta*forcing, q);
          else
            phi.submit_value(parameters.theta*forcing, q);
        }
        //--- Stages of TR-BDF2
        else {
          //--- First stage of TR_BDF2
          if(TR_BDF2_stage == 1) {
            phi.submit_gradient(parameters.theta*flux, q);
            if(parameters.is_stationary == false)
              phi.submit_value(-1.0/parameters.time_step*W + parameters.theta*forcing, q);
            else
              phi.submit_value(parameters.theta*forcing, q);
          }
          //--- Second stage of TR_BDF2
          else if(TR_BDF2_stage == 2) {
            phi.submit_gradient(gamma_2*flux, q);
            if(parameters.is_stationary == false)
              phi.submit_value(-1.0/parameters.time_step*W + gamma_2*forcing, q);
            else
              phi.submit_value(gamma_2*forcing, q);
          }
        }
      }
      phi.integrate(true, true);
      //--- Assemble cell-matrix (the integrated values are stored internally and should
      //--- be accessible throguh get_dof_value without using another temporary vector)
      for(unsigned int i = 0; i < dofs_per_component; ++i) {
        const auto& tmp_dof_getter = phi.get_dof_value(i);
        for(unsigned int k = 0; k < EulerEquations<dim>::n_components; ++k) {
          for(unsigned int j = 0; j < dofs_per_cell; ++j) {
            cell_matrix(EulerEquations<dim>::n_components*i + k, j) =
            tmp_dof_getter[EulerEquations<dim>::n_components*i + k].fastAccessDx(j);
          }
        }
      }
      //--- Assemble rhs extracting the degrees of freedom from src
      local_dof_indices = phi.local_dof_indices;
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        cell_src(i) = src(local_dof_indices[i]);
      //--- Execute matrix-vector product and save to dts
      cell_matrix.vmult(cell_dst, cell_src);
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        dst(local_dof_indices[i]) += cell_dst(i);
    }
  }

  // Assemble matrix-vector for interior faces
  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void ConservationLawOperator<dim, fe_degree, n_q_points_1d, Number>::
  assemble_face_term(const MatrixFree<dim, Number>&                    data,
                     LinearAlgebra::distributed::Vector<Number>&       dst,
                     const LinearAlgebra::distributed::Vector<Number>& src,
                     const std::pair<unsigned int, unsigned int>&      face_range) {
    Tensor<1, EulerEquations<dim>::n_components, VectorizedArray<Sacado::Fad::DFad<Number>>> normal_fluxes;

    FEFaceEvaluation<dim, fe_degree, n_q_points_1d, EulerEquations<dim>::n_components, Sacado::Fad::DFad<Number>> phi_m(data, false);
    FEFaceEvaluation<dim, fe_degree, n_q_points_1d, EulerEquations<dim>::n_components, Sacado::Fad::DFad<Number>> phi_p(data, true);

    const unsigned int dofs_per_cell      = data.get_dofs_per_cell();
    const unsigned int dofs_per_component = dofs_per_cell/EulerEquations<dim>::n_components;

    std::vector<unsigned int> local_dof_indices(dofs_per_cell);
    FullMatrix<double>        face_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>            face_src(dofs_per_cell), face_dst(dofs_per_cell);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_p.reinit(face);
      phi_p.gather_evaluate(src, true, false);

      phi_m.reinit(face);
      phi_m.gather_evaluate(src, true, false);

      for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
        const auto Wminus = phi_m.get_value(q);
        const auto Wplus  = phi_p.get_value(q);
        EulerEquations<dim>::numerical_normal_flux(phi_p.get_normal_vector(q), Wplus, Wminus, lambda_old, normal_fluxes);
        if(parameters.time_integration_scheme == "Theta_Method") {
          phi_m.submit_value(parameters.theta*normal_fluxes, q);
          phi_p.submit_value(-parameters.theta*normal_fluxes, q);
        }
        else {
          if(TR_BDF2_stage == 1) {
            phi_m.submit_value(parameters.theta*normal_fluxes, q);
            phi_p.submit_value(-parameters.theta*normal_fluxes, q);
          }
          else {
            phi_m.submit_value(gamma_2*normal_fluxes, q);
            phi_p.submit_value(-gamma_2*normal_fluxes, q);
          }
        }
      }
      phi_p.integrate(true, false);
      phi_m.integrate(true, false);
      //--- Assemble face-matrix (the integrated values are stored internally and should
      //--- be accessible throguh get_dof_value without using another temporary vector)
      for(unsigned int i = 0; i < dofs_per_component; ++i) {
        const auto& tmp_dof_getter = phi_p.get_dof_value(i);
        for(unsigned int k = 0; k < EulerEquations<dim>::n_components; ++k) {
          for(unsigned int j = 0; j < dofs_per_cell; ++j) {
            face_matrix(EulerEquations<dim>::n_components*i + k, j) =
            tmp_dof_getter[EulerEquations<dim>::n_components*i + k].fastAccessDx(j);
          }
        }
      }
      //--- Assemble rhs extracting the degrees of freedom from src
      local_dof_indices = phi_p.local_dof_indices;
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        face_src(i) = src(local_dof_indices[i]);
      //--- Execute matrix-vector product and save to dts
      face_matrix.vmult(face_dst, face_src);
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        dst(local_dof_indices[i]) += face_dst(i);
      //--- Repeat the same procedure using neighbor dofs per cell (we extract src from neighbor FEFaceEvaluation)
      for(unsigned int i = 0; i < dofs_per_component; ++i) {
        const auto& tmp_dof_getter = phi_p.get_dof_value(i);
        for(unsigned int k = 0; k < EulerEquations<dim>::n_components; ++k) {
          for(unsigned int j = 0; j < dofs_per_cell; ++j) {
            face_matrix(EulerEquations<dim>::n_components*i + k, j) =
            tmp_dof_getter[EulerEquations<dim>::n_components*i + k].fastAccessDx(dofs_per_cell + j);
          }
        }
      }
      local_dof_indices = phi_m.local_dof_indices;
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        face_src(i) = src(local_dof_indices[i]);
      face_matrix.vmult(face_dst, face_src);
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        dst(local_dof_indices[i]) += face_dst(i);

      //--- Repeat the whole procedure swapping the roles of the two FEFaceEvaluation
      for(unsigned int i = 0; i < dofs_per_component; ++i) {
        const auto& tmp_dof_getter = phi_m.get_dof_value(i);
        for(unsigned int k = 0; k < EulerEquations<dim>::n_components; ++k) {
          for(unsigned int j = 0; j < dofs_per_cell; ++j) {
            face_matrix(EulerEquations<dim>::n_components*i + k, j) =
            tmp_dof_getter[EulerEquations<dim>::n_components*i + k].fastAccessDx(j);
          }
        }
      }
      local_dof_indices = phi_m.local_dof_indices;
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        face_src(i) = src(local_dof_indices[i]);
      face_matrix.vmult(face_dst, face_src);
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        dst(local_dof_indices[i]) += face_dst(i);
      for(unsigned int i = 0; i < dofs_per_component; ++i) {
        const auto& tmp_dof_getter = phi_m.get_dof_value(i);
        for(unsigned int k = 0; k < EulerEquations<dim>::n_components; ++k) {
          for(unsigned int j = 0; j < dofs_per_cell; ++j) {
            face_matrix(EulerEquations<dim>::n_components*i + k, j) =
            tmp_dof_getter[EulerEquations<dim>::n_components*i + k].fastAccessDx(dofs_per_cell + j);
          }
        }
      }
      local_dof_indices = phi_p.local_dof_indices;
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        face_src(i) = src(local_dof_indices[i]);
      face_matrix.vmult(face_dst, face_src);
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        dst(local_dof_indices[i]) += face_dst(i);
    }
  }

  // Collect all the contributions and execute loop
  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void ConservationLawOperator<dim, fe_degree, n_q_points_1d, Number>::
  vmult(LinearAlgebra::distributed::Vector<Number>&       dst,
        const LinearAlgebra::distributed::Vector<Number>& src) const {

    this->data->loop(&ConservationLawOperator::assemble_cell_term,
                     &ConservationLawOperator::assemble_face_term,
                     &ConservationLawOperator::assemble_boundary_term,
                     this,
                     dst,
                     src,
                     true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  // Assemble matrix-vector for boundary faces
  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void ConservationLawOperator<dim, fe_degree, n_q_points_1d, Number>::
  assemble_boundary_term(const MatrixFree<dim, Number>&                    data,
                         LinearAlgebra::distributed::Vector<Number>&       dst,
                         const LinearAlgebra::distributed::Vector<Number>& src,
                         const std::pair<unsigned int, unsigned int>&      face_range) {
    Tensor<1, EulerEquations<dim>::n_components, VectorizedArray<Sacado::Fad::DFad<Number>>> normal_fluxes;
    Tensor<1, EulerEquations<dim>::n_components, VectorizedArray<Sacado::Fad::DFad<Number>>> Wminus;

    const unsigned int dofs_per_cell      = data.get_dofs_per_cell();
    const unsigned int dofs_per_component = dofs_per_cell/EulerEquations<dim>::n_components;

    std::vector<unsigned int> local_dof_indices(dofs_per_cell);
    FullMatrix<double>        face_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>            face_src(dofs_per_cell), face_dst(dofs_per_cell);

    FEFaceEvaluation<dim, fe_degree, n_q_points_1d, EulerEquations<dim>::n_components, Sacado::Fad::DFad<Number>> phi(data, true);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi.reinit(face);
      phi.gather_evaluate(src, true, false);

      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto Wplus = phi.get_value(q);
        const auto boundary_id = data.get_boundary_id(face);
        Assert(boundary_id < Parameters::AllParameters<dim>::max_n_boundaries,
               ExcIndexRange(boundary_id, 0, Parameters::AllParameters<dim>::max_n_boundaries));
        Vector<double> boundary_values(EulerEquations<dim>::n_components);
        const auto& p_vectorized = phi.quadrature_point(q);
        Point<dim> p;
        for(unsigned d = 0; d < dim; ++d)
          p[d] = p_vectorized[d][0];
        if(parameters.testcase == 0)
          parameters.boundary_conditions[boundary_id].values.vector_value(p, boundary_values);
        else
          parameters.exact_solution.vector_value(p, boundary_values);
        EulerEquations<dim>::compute_Wminus(parameters.boundary_conditions[boundary_id].kind,
                                            phi.get_normal_vector(q), Wplus, boundary_values, Wminus);
        EulerEquations<dim>::numerical_normal_flux(phi.get_normal_vector(q), Wplus, Wminus, lambda_old, normal_fluxes);
        if(parameters.time_integration_scheme == "Theta_Method")
          phi.submit_value(-parameters.theta*normal_fluxes, q);
        else {
          if(TR_BDF2_stage == 1)
            phi.submit_value(-parameters.theta*normal_fluxes, q);
          else
            phi.submit_value(-gamma_2*normal_fluxes, q);
        }
      }
      phi.integrate(true, false);
      //--- Assemble face-matrix (the integrated values are stored internally and should
      //--- be accessible throguh get_dof_value without using another temporary vector)
      for(unsigned int i = 0; i < dofs_per_component; ++i) {
        const auto& tmp_dof_getter = phi.get_dof_value(i);
        for(unsigned int k = 0; k < EulerEquations<dim>::n_components; ++k) {
          for(unsigned int j = 0; j < dofs_per_cell; ++j) {
            face_matrix(EulerEquations<dim>::n_components*i + k, j) =
            tmp_dof_getter[EulerEquations<dim>::n_components*i + k].fastAccessDx(j);
          }
        }
      }
      //--- Assemble rhs extracting the degrees of freedom from src
      local_dof_indices = phi.local_dof_indices;
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        face_src(i) = src(local_dof_indices[i]);
      //--- Execute matrix-vector product and save to dts
      face_matrix.vmult(face_dst, face_src);
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        dst(local_dof_indices[i]) += face_dst(i);
    }
  }


  // @sect3{Conservation law class}

  // Here finally comes the class that actually does something with all the
  // Euler equation and parameter specifics we've defined above.
  //
  template<int dim>
  class ConservationLaw {
  public:
    ConservationLaw(ParameterHandler& prm);
    void run();

  private:
    //--- Function to build the grid and distribute dofs
    void make_grid_and_dofs();

    void setup_system();

    //--- MatrixFree part
    void assemble_explicit_system(const double current_time);
    void assemble_rhs(const double next_time);

    std::pair<unsigned int, double> solve(LinearAlgebra::distributed::Vector<double>& solution);

    void compute_refinement_indicators(Vector<double>& indicator) const;
    void refine_grid(const Vector<double>& indicator);

    void compute_errors();

    void output_results() const;

    using SystemMatrixType = ConservationLawOperator<dim, EulerEquations<dim>::fe_degree,
                                                     EulerEquations<dim>::n_q_points_1d, double>;
    SystemMatrixType matrix_free;

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

    LinearAlgebra::ReadWriteVector<double> newton_update_exchanger;
    LinearAlgebra::distributed::Vector<double> newton_update_mf;
    LinearAlgebra::distributed::Vector<double> right_hand_side, right_hand_side_explicit, right_hand_side_mf;

    LA::MPI::Vector locally_relevant_solution;  //--- Extra variables for parallel purposes (read-only)

    double             gamma_2;       //--- Extra parameter of TR_BDF2
    double             gamma_3;       //--- Extra parameter of TR_BDF2
    unsigned int       TR_BDF2_stage; //--- Extra parameter for counting stage TR_BDF2
    ConditionalOStream pcout;         //--- Extra parameter for parallel cout
    ConditionalOStream verbose_cout;
    std::ofstream      output_niter;
    std::ofstream      output_error;
    std::ofstream      time_out;     //--- Auxiliary ofstream for time output
    ConditionalOStream ptime_out;    //--- Auxiliary conditional stream for time output
    TimerOutput        time_table;   //--- Auxiliary Table for time
  };


  // @sect4{ConservationLaw::ConservationLaw}
  //
  // There is nothing much to say about the constructor. Essentially, it reads
  // the input file and fills the parameter object with the parsed values:
  template<int dim>
  ConservationLaw<dim>::ConservationLaw(ParameterHandler& prm):
      matrix_free(prm),
      parameters(prm),
      communicator(MPI_COMM_WORLD),
      triangulation(communicator),
      mapping(),
      fe(FE_DGQ<dim>(EulerEquations<dim>::fe_degree), EulerEquations<dim>::n_components),
      dof_handler(triangulation),
      quadrature(fe.degree + 1),
      face_quadrature(fe.degree + 1),
      pcout(std::cout, Utilities::MPI::this_mpi_process(communicator) == 0),
      verbose_cout(std::cout, Utilities::MPI::this_mpi_process(communicator) == 0 &&
                              parameters.output == Parameters::Solver::verbose),
      output_niter("./" + parameters.dir + "/linear_number_iterations.dat", std::ofstream::out),
      output_error("./" + parameters.dir + "/error_analysis.dat", std::ofstream::out),
      time_out("./" + parameters.dir + "/time_analysis_" +
               Utilities::int_to_string(Utilities::MPI::n_mpi_processes(communicator)) + "proc.dat"),
      ptime_out(time_out, Utilities::MPI::this_mpi_process(communicator) == 0),
      time_table(ptime_out, TimerOutput::never, TimerOutput::wall_times) {
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
    newton_update_exchanger.reinit(locally_owned_dofs, communicator);
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

    matrix_free.clear();
    const AffineConstraints<double> dummy;
    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                            update_quadrature_points | update_values);
    additional_data.mapping_update_flags_inner_faces = (update_JxW_values | update_quadrature_points |
                                                        update_normal_vectors | update_values);
    additional_data.mapping_update_flags_boundary_faces = (update_JxW_values | update_quadrature_points |
                                                           update_normal_vectors | update_values);
    additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
    std::shared_ptr<MatrixFree<dim, double>> matrix_free_storage(new MatrixFree<dim, double>());
    matrix_free_storage->reinit(mapping, dof_handler, dummy, face_quadrature, additional_data);
    matrix_free.initialize(matrix_free_storage);
    matrix_free.initialize_dof_vector(right_hand_side);
    matrix_free.initialize_dof_vector(right_hand_side_explicit);
    matrix_free.initialize_dof_vector(right_hand_side_mf);
    matrix_free.initialize_dof_vector(newton_update_mf);

    //--- Read-only variant of the solution that must be set after the solution
    locally_relevant_solution.reinit(locally_owned_dofs, locally_relevant_dofs, communicator);
  }


  // @sect4{ConservationLaw::assemble_explicit_system}
  //
  template<int dim>
  void ConservationLaw<dim>::assemble_explicit_system(const double current_time) {
    TimerOutput::Scope t(time_table, "Assemble explicit term");

    LinearAlgebra::ReadWriteVector<double> old_solution_tmp, intermediate_solution_tmp;
    old_solution_tmp.reinit(old_solution);
    intermediate_solution_tmp.reinit(intermediate_solution);
    LinearAlgebra::distributed::Vector<double> old_solution_tmp1, intermediate_solution_tmp1;
    matrix_free.initialize_dof_vector(old_solution_tmp1);
    matrix_free.initialize_dof_vector(intermediate_solution_tmp1);
    old_solution_tmp1.import(old_solution_tmp, VectorOperation::insert);
    intermediate_solution_tmp1.import(intermediate_solution_tmp, VectorOperation::insert);

    matrix_free.vmult_explicit(current_time, {old_solution_tmp1, intermediate_solution_tmp1}, right_hand_side_explicit);
  }


  // @sect4{ConservationLaw::assemble_rhs}
  //
  template<int dim>
  void ConservationLaw<dim>::assemble_rhs(const double next_time) {
    TimerOutput::Scope t(time_table, "Assemble rhs");

    LinearAlgebra::ReadWriteVector<double> current_solution_tmp;
    current_solution_tmp.reinit(current_solution);
    LinearAlgebra::distributed::Vector<double> current_solution_tmp1;
    matrix_free.initialize_dof_vector(current_solution_tmp1);
    current_solution_tmp1.import(current_solution_tmp, VectorOperation::insert);

    matrix_free.vmult_rhs(next_time, current_solution_tmp1, right_hand_side_mf);
  }


  // @sect4{ConservationLaw::solve}
  //
  // Here, we actually solve the linear system, using either of Trilinos'
  // Aztec or Amesos linear solvers. The result of the computation will be
  // written into the argument vector passed to this function. The result is a
  // pair of number of iterations and the final linear residual.

  template<int dim>
  std::pair<unsigned int, double>
  ConservationLaw<dim>::solve(LinearAlgebra::distributed::Vector<double>& newton_update) {
    TimerOutput::Scope t(time_table, "Solve");

    switch(parameters.solver) {
      case Parameters::Solver::direct:
      {
        SolverControl  solver_control(1, 0);
        TrilinosWrappers::SolverDirect::AdditionalData data(parameters.output == Parameters::Solver::verbose);
        TrilinosWrappers::SolverDirect direct(solver_control, data);

        direct.solve(newton_update, right_hand_side);

        return {solver_control.last_step(), solver_control.last_value()};
      }

      case Parameters::Solver::gmres:
      {
        /*
        Epetra_Vector x(View, system_matrix.trilinos_matrix().DomainMap(),
                        newton_update.begin());
        Epetra_Vector b(View, system_matrix.trilinos_matrix().RangeMap(),
                        right_hand_side.begin());

        AztecOO solver;
        solver.SetAztecOption(AZ_output,
                              (parameters.output == Parameters::Solver::quiet ?
                               AZ_none : AZ_all));
        solver.SetAztecOption(AZ_solver, AZ_gmres);
        solver.SetRHS(&b);
        solver.SetLHS(&x);

        solver.SetAztecOption(AZ_precond, AZ_dom_decomp);
        solver.SetAztecOption(AZ_subdomain_solve, AZ_ilut);
        solver.SetAztecOption(AZ_overlap, 0);
        solver.SetAztecOption(AZ_reorder, 0);

        solver.SetAztecParam(AZ_drop, parameters.ilut_drop);
        solver.SetAztecParam(AZ_ilut_fill, parameters.ilut_fill);
        solver.SetAztecParam(AZ_athresh, parameters.ilut_atol);
        solver.SetAztecParam(AZ_rthresh, parameters.ilut_rtol);

        solver.SetUserMatrix(const_cast<Epetra_CrsMatrix*>(&system_matrix.trilinos_matrix()));

        solver.Iterate(parameters.max_iterations, parameters.linear_residual);
        */
        SolverControl solver_control(parameters.max_iterations, parameters.linear_residual);
        SolverGMRES<LinearAlgebra::distributed::Vector<double>> gmres(solver_control);
        gmres.solve(matrix_free, newton_update, right_hand_side, PreconditionIdentity());

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
    /*
    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(EulerEquations<dim>::first_momentum_component,
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
                                                                    VectorTools::L2_norm);
    */
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

    TimerOutput::Scope t(time_table, "Output results");
    output_results();

    // We then enter into the main time stepping loop. At the top we simply
    // output some status information so one can keep track of where a
    // computation is, as well as the header for a table that indicates
    // progress of the nonlinear inner iteration:
    LA::MPI::Vector newton_update;
    newton_update.reinit(locally_owned_dofs, communicator);

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
      //--- Restyling for TR_BDF2
      if(parameters.time_integration_scheme == "Theta_Method") {
        assemble_explicit_system(time);
        while(true) {
          if(parameters.testcase == 1)
            parameters.exact_solution.set_time(time + parameters.time_step);
          locally_relevant_solution = current_solution;
          assemble_rhs(time + parameters.time_step);
          right_hand_side = right_hand_side_explicit;
          right_hand_side.import(right_hand_side_mf, VectorOperation::add);

          const double res_norm = right_hand_side.l2_norm();
          if(std::fabs(res_norm) < 1e-10) {
            if(Utilities::MPI::this_mpi_process(communicator) == 0)
              std::printf("   %-16.3e (converged)\n\n", res_norm);
            break;
          }
          else {
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
        assemble_explicit_system(time);
        //--- First stage Newton loop
        while(true) {
          if(parameters.testcase == 1)
            parameters.exact_solution.set_time(time + 2.0*parameters.theta*parameters.time_step);
          locally_relevant_solution = current_solution;
          assemble_rhs(time + 2.0*parameters.theta*parameters.time_step);
          right_hand_side = right_hand_side_explicit;
          right_hand_side.import(right_hand_side_mf, VectorOperation::add);

          const double res_norm = right_hand_side.l2_norm();
          if(std::fabs(res_norm) < 1e-10) {
            if(Utilities::MPI::this_mpi_process(communicator) == 0)
              std::printf("   %-16.3e (converged)\n\n", res_norm);
            break;
          }
          else {
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
        matrix_free.set_TR_BDF2_stage(TR_BDF2_stage);
        assemble_explicit_system(time);
        //--- TR_BDF2 second stage Newton loop
        while(true) {
          if(parameters.testcase == 1)
            parameters.exact_solution.set_time(time + parameters.time_step);
          locally_relevant_solution = current_solution;
          assemble_rhs(time + parameters.time_step);
          right_hand_side = right_hand_side_explicit;
          right_hand_side.import(right_hand_side_mf, VectorOperation::add);

          const double res_norm = right_hand_side.l2_norm();
          if(std::fabs(res_norm) < 1e-10) {
            if(Utilities::MPI::this_mpi_process(communicator) == 0)
              std::printf("   %-16.3e (converged)\n\n", res_norm);
            break;
          }
          else {
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
        matrix_free.set_TR_BDF2_stage(TR_BDF2_stage);
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
      if(parameters.testcase == 1)
        compute_errors();

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
    const unsigned int dim = 2;
    Parameters::AllParameters<dim>::declare_parameters(prm);
    prm.parse_input(argv[1]);

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, dealii::numbers::invalid_unsigned_int);

    ConservationLaw<dim> cons(prm);
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

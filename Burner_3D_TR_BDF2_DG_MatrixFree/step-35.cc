// @sect3{Include files}

// We start by including all the necessary deal.II header files and some C++
// related ones.
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/parallel.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <cmath>
#include <iostream>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/fe/component_mask.h>

#include <deal.II/base/timer.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

// Finally this is as in all previous programs:
namespace Step35 {
  using namespace dealii;

  // @sect3{Run time parameters}
  //
  // Since our method has several parameters that can be fine-tuned we put them
  // into an external file, so that they can be determined at run-time.
  //
  namespace RunTimeParameters {
    class Data_Storage {
    public:
      Data_Storage();

      void read_data(const std::string& filename);

      double initial_time;
      double final_time;

      double Reynolds;
      double dt;

      unsigned int max_loc_refinements;
      unsigned int min_loc_refinements;

      unsigned int vel_max_iterations;
      unsigned int vel_Krylov_size;
      unsigned int vel_off_diagonals;
      unsigned int vel_update_prec;
      double       vel_eps;
      double       vel_diag_strength;

      bool         verbose;
      unsigned int output_interval;

      std::string dir;

      unsigned int refinement_iterations;

    protected:
      ParameterHandler prm;
    };

    // In the constructor of this class we declare all the parameters.
    Data_Storage::Data_Storage(): initial_time(0.0),
                                  final_time(1.0),
                                  Reynolds(1.0),
                                  dt(5e-4),
                                  max_loc_refinements(0),
                                  min_loc_refinements(0),
                                  vel_max_iterations(1000),
                                  vel_Krylov_size(30),
                                  vel_off_diagonals(60),
                                  vel_update_prec(15),
                                  vel_eps(1e-12),
                                  vel_diag_strength(0.01),
                                  verbose(true),
                                  output_interval(15),
                                  refinement_iterations(0) {
      prm.enter_subsection("Physical data");
      {
        prm.declare_entry("initial_time",
                          "0.",
                          Patterns::Double(0.0),
                          " The initial time of the simulation. ");
        prm.declare_entry("final_time",
                          "1.",
                          Patterns::Double(0.0),
                          " The final time of the simulation. ");
        prm.declare_entry("Reynolds",
                          "1.",
                          Patterns::Double(0.0),
                          " The Reynolds number. ");
      }
      prm.leave_subsection();

      prm.enter_subsection("Time step data");
      {
        prm.declare_entry("dt",
                          "5e-4",
                          Patterns::Double(0.0),
                          " The time step size. ");
      }
      prm.leave_subsection();

      prm.enter_subsection("Space discretization");
      {
        prm.declare_entry("max_loc_refinements",
                          "4",
                           Patterns::Integer(1, 10),
                           " The number of maximum local refinements. ");
        prm.declare_entry("min_loc_refinements",
                          "2",
                           Patterns::Integer(1, 10),
                           " The number of minimum local refinements. ");
      }
      prm.leave_subsection();

      prm.enter_subsection("Data solve velocity");
      {
        prm.declare_entry("max_iterations",
                          "1000",
                          Patterns::Integer(1, 30000),
                          " The maximal number of iterations GMRES must make. ");
        prm.declare_entry("eps",
                          "1e-12",
                          Patterns::Double(0.0),
                          " The stopping criterion. ");
        prm.declare_entry("Krylov_size",
                          "30",
                          Patterns::Integer(1),
                          " The size of the Krylov subspace to be used. ");
        prm.declare_entry("off_diagonals",
                          "60",
                          Patterns::Integer(0),
                          " The number of off-diagonal elements ILU must "
                          "compute. ");
        prm.declare_entry("diag_strength",
                          "0.01",
                          Patterns::Double(0.0),
                          " Diagonal strengthening coefficient. ");
        prm.declare_entry("update_prec",
                          "15",
                          Patterns::Integer(1),
                          " This number indicates how often we need to "
                          "update the preconditioner");
      }
      prm.leave_subsection();

      prm.declare_entry("refinement_iterations",
                        "0",
                        Patterns::Integer(0),
                        " This number indicates how often we need to "
                        "refine the mesh");

      prm.declare_entry("saving directory", "SimTest");

      prm.declare_entry("verbose",
                        "true",
                        Patterns::Bool(),
                        " This indicates whether the output of the solution "
                        "process should be verbose. ");

      prm.declare_entry("output_interval",
                        "1",
                        Patterns::Integer(1),
                        " This indicates between how many time steps we print "
                        "the solution. ");
    }

    // Function to read all declared parameters in the constructor
    void Data_Storage::read_data(const std::string& filename) {
      std::ifstream file(filename);
      AssertThrow(file, ExcFileNotOpen(filename));

      prm.parse_input(file);

      prm.enter_subsection("Physical data");
      {
        initial_time = prm.get_double("initial_time");
        final_time   = prm.get_double("final_time");
        Reynolds     = prm.get_double("Reynolds");
      }
      prm.leave_subsection();

      prm.enter_subsection("Time step data");
      {
        dt = prm.get_double("dt");
      }
      prm.leave_subsection();

      prm.enter_subsection("Space discretization");
      {
        max_loc_refinements = prm.get_integer("max_loc_refinements");
        min_loc_refinements = prm.get_integer("min_loc_refinements");
      }
      prm.leave_subsection();

      prm.enter_subsection("Data solve velocity");
      {
        vel_max_iterations = prm.get_integer("max_iterations");
        vel_eps            = prm.get_double("eps");
        vel_Krylov_size    = prm.get_integer("Krylov_size");
        vel_off_diagonals  = prm.get_integer("off_diagonals");
        vel_diag_strength  = prm.get_double("diag_strength");
        vel_update_prec    = prm.get_integer("update_prec");
      }
      prm.leave_subsection();

      dir = prm.get("saving directory");

      refinement_iterations = prm.get_integer("refinement_iterations");

      verbose = prm.get_bool("verbose");

      output_interval = prm.get_integer("output_interval");
    }

  } // namespace RunTimeParameters


  // @sect3{Equation data}

  // In the next namespace, we declare the initial and boundary conditions
  //
  namespace EquationData {
    static const unsigned int degree_p = 1;

    // With this class defined, we declare class that describes the boundary
    // conditions for velocity:
    template<int dim>
    class Velocity: public Function<dim> {
    public:
      Velocity(const double initial_time = 0.0);

      virtual double value(const Point<dim>&  p,
                           const unsigned int component = 0) const override;

      virtual void vector_value(const Point<dim>& p,
                                Vector<double>&   values) const override;
    };


    template<int dim>
    Velocity<dim>::Velocity(const double initial_time): Function<dim>(dim, initial_time) {}


    template<int dim>
    double Velocity<dim>::value(const Point<dim>& p, const unsigned int component) const {
      AssertIndexRange(component, 3);
      if(component == 1)
        return -24.3125555326;
      else
        return 0.0;
    }


    template<int dim>
    void Velocity<dim>::vector_value(const Point<dim>& p, Vector<double>& values) const {
      Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));
      for(unsigned int i = 0; i < dim; ++i)
        values[i] = value(p, i);
    }



    // We do the same for the pressure (since it is a scalar field) we can derive
    // directly from the deal.II built-in class Function
    template<int dim>
    class Pressure: public Function<dim> {
    public:
      Pressure(const double initial_time = 0.0);

      virtual double value(const Point<dim>&  p,
                           const unsigned int component = 0) const override;
    };


    template<int dim>
    Pressure<dim>::Pressure(const double initial_time): Function<dim>(1, initial_time) {}


    template<int dim>
    double Pressure<dim>::value(const Point<dim>&  p, const unsigned int component) const {
      (void)component;
      AssertIndexRange(component, 1);
      return 1.0;
    }

  } // namespace EquationData



  // Class for post-processing vorticity
  //
  template<int dim>
  class PostprocessorVorticity: public DataPostprocessor<dim> {
  public:
    virtual void evaluate_vector_field(const DataPostprocessorInputs::Vector<dim>& inputs,
                                       std::vector<Vector<double>>&                computed_quantities) const override;

    virtual std::vector<std::string> get_names() const override;

    virtual std::vector<DataComponentInterpretation::DataComponentInterpretation> get_data_component_interpretation() const override;

    virtual UpdateFlags get_needed_update_flags() const override;
  };


  template <int dim>
  void PostprocessorVorticity<dim>::evaluate_vector_field(const DataPostprocessorInputs::Vector<dim>& inputs,
                                                          std::vector<Vector<double>>&                computed_quantities) const {
    const unsigned int n_quadrature_points = inputs.solution_values.size();

    Assert(inputs.solution_gradients.size() == n_quadrature_points, ExcInternalError());
    Assert(computed_quantities.size() == n_quadrature_points, ExcInternalError());

    Assert(inputs.solution_values[0].size() == dim, ExcInternalError());

    if(dim == 2) {
      Assert(computed_quantities[0].size() == 1, ExcInternalError());
    }
    else {
      Assert(computed_quantities[0].size() == dim, ExcInternalError());
    }

    if(dim == 2) {
      for(unsigned int q = 0; q < n_quadrature_points; ++q)
        computed_quantities[q](0) = inputs.solution_gradients[q][1][0] - inputs.solution_gradients[q][0][1];
    }
    else {
      for(unsigned int q = 0; q < n_quadrature_points; ++q) {
        computed_quantities[q](0) = inputs.solution_gradients[q][2][1] - inputs.solution_gradients[q][1][2];
        computed_quantities[q](1) = inputs.solution_gradients[q][0][2] - inputs.solution_gradients[q][2][0];
        computed_quantities[q](2) = inputs.solution_gradients[q][1][0] - inputs.solution_gradients[q][0][1];
      }
    }
  }


  template<int dim>
  std::vector<std::string> PostprocessorVorticity<dim>::get_names() const {
    std::vector<std::string> names;
    names.emplace_back("vorticity");
    if(dim == 3) {
      names.emplace_back("vorticity");
      names.emplace_back("vorticity");
    }

    return names;
  }


  template<int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  PostprocessorVorticity<dim>::get_data_component_interpretation() const {
    std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation;
    if(dim == 2)
      interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    else {
      interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
      interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
      interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    }

    return interpretation;
  }


  template<int dim>
  UpdateFlags PostprocessorVorticity<dim>::get_needed_update_flags() const {
    return update_gradients;
  }



  // @sect3{ <code>NavierStokesProjectionOperator::NavierStokesProjectionOperator</code> }
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  class NavierStokesProjectionOperator: public MatrixFreeOperators::Base<dim, Vec> {
  public:
    NavierStokesProjectionOperator();

    NavierStokesProjectionOperator(RunTimeParameters::Data_Storage& data);

    void set_dt(const double time_step);

    void set_Reynolds(const double Reynolds);

    void set_TR_BDF2_stage(const unsigned int stage);

    void set_NS_stage(const unsigned int stage);

    void set_u_extr(const Vec& src);

    void vmult_rhs_velocity(Vec& dst, const std::vector<Vec>& src) const;

    void vmult_rhs_pressure(Vec& dst, const std::vector<Vec>& src) const;

    void vmult_grad_p_projection(Vec& dst, const Vec& src) const;

    virtual void compute_diagonal() override;

  protected:
    double       Re;
    double       dt;

    double       gamma;
    double       a31;
    double       a32;
    double       a33;

    unsigned int TR_BDF2_stage;
    unsigned int NS_stage;

    virtual void apply_add(Vec& dst, const Vec& src) const override;

  private:
    const double a21 = 0.5;
    const double a22 = 0.5;

    const double theta_v = 1.0;
    const double theta_p = 1.0;
    const double C_p = 1.0*(fe_degree_p + 1)*(fe_degree_p + 1);
    const double C_u = 1.0*(fe_degree_v + 1)*(fe_degree_v + 1);

    Vec                         u_extr;

    EquationData::Velocity<dim> vel_boundary;

    void assemble_rhs_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_velocity(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                             Vec&                                         dst,
                                             const std::vector<Vec>&                      src,
                                             const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_rhs_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                             Vec&                                         dst,
                                             const std::vector<Vec>&                      src,
                                             const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_face_term_velocity(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const Vec&                                   src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const Vec&                                   src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const {}

    void assemble_cell_term_projection_grad_p(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const Vec&                                   src,
                                              const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_cell_term_projection_grad_p(const MatrixFree<dim, Number>&               data,
                                                  Vec&                                         dst,
                                                  const Vec&                                   src,
                                                  const std::pair<unsigned int, unsigned int>& cell_range) const;

    void assemble_diagonal_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const Vec&                                   src,
                                              const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_diagonal_face_term_velocity(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const Vec&                                   src,
                                              const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_diagonal_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                                  Vec&                                         dst,
                                                  const Vec&                                   src,
                                                  const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_diagonal_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const Vec&                                   src,
                                              const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_diagonal_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const Vec&                                   src,
                                              const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_diagonal_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                                  Vec&                                         dst,
                                                  const Vec&                                   src,
                                                  const std::pair<unsigned int, unsigned int>& face_range) const {}
  };


  // Default constructor
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  NavierStokesProjectionOperator():
    MatrixFreeOperators::Base<dim, Vec>(), Re(), dt(), gamma(2.0 - std::sqrt(2.0)),
                                           a31((1.0 - gamma)/(2.0*(2.0 - gamma))), a32(a31), a33(1.0/(2.0 - gamma)),
                                           TR_BDF2_stage(1), NS_stage(1), u_extr() {}


  // Constructor with runtime parameters storage
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  NavierStokesProjectionOperator(RunTimeParameters::Data_Storage& data):
    MatrixFreeOperators::Base<dim, Vec>(), Re(data.Reynolds), dt(data.dt),
                                           gamma(2.0 - std::sqrt(2.0)), a31((1.0 - gamma)/(2.0*(2.0 - gamma))),
                                           a32(a31), a33(1.0/(2.0 - gamma)), TR_BDF2_stage(1), NS_stage(1), u_extr(),
                                           vel_boundary(data.initial_time) {}


  // Setter of time-step
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  set_dt(const double time_step) {
    dt = time_step;
  }


  // Setter of Reynolds number
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  set_Reynolds(const double Reynolds) {
    Re = Reynolds;
  }


  // Setter of TR-BDF2 stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  set_TR_BDF2_stage(const unsigned int stage) {
    AssertIndexRange(stage, 3);
    Assert(stage > 0, ExcInternalError());
    TR_BDF2_stage = stage;
  }


  // Setter of NS stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  set_NS_stage(const unsigned int stage) {
    AssertIndexRange(stage, 4);
    Assert(stage > 0, ExcInternalError());
    NS_stage = stage;
  }


  // Setter of extrapolated velocity for different stages
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  set_u_extr(const Vec& src) {
    u_extr = src;
    u_extr.update_ghost_values();
  }


  // Assemble rhs cell term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  assemble_rhs_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(TR_BDF2_stage == 1) {
      FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0), phi_old(data, 0), phi_old_extr(data, 0);
      FEEvaluation<dim, fe_degree_p, n_q_points_1d_v, 1, Number> phi_old_press(data, 1);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_old.reinit(cell);
        phi_old.gather_evaluate(src[0], true, true);
        phi_old_extr.reinit(cell);
        phi_old_extr.gather_evaluate(src[1], true, false);
        phi_old_press.reinit(cell);
        phi_old_press.gather_evaluate(src[2], true, false);
        phi.reinit(cell);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& u_n                = phi_old.get_value(q);
          const auto& grad_u_n           = phi_old.get_gradient(q);
          const auto& u_n_gamma_ov_2     = phi_old_extr.get_value(q);
          const auto& tensor_product_u_n = outer_product(u_n, u_n_gamma_ov_2);
          const auto& p_n                = phi_old_press.get_value(q);
          auto p_n_times_identity        = tensor_product_u_n;
          p_n_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d)
            p_n_times_identity[d][d] = p_n;
          phi.submit_value(1.0/(gamma*dt)*u_n, q);
          phi.submit_gradient(-a21/Re*grad_u_n + a21*tensor_product_u_n + p_n_times_identity, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0), phi_old(data, 0), phi_int(data, 0);
      FEEvaluation<dim, fe_degree_p, n_q_points_1d_v, 1, Number> phi_old_press(data, 1);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_old.reinit(cell);
        phi_old.gather_evaluate(src[0], true, true);
        phi_int.reinit(cell);
        phi_int.gather_evaluate(src[1], true, true);
        phi_old_press.reinit(cell);
        phi_old_press.gather_evaluate(src[2], true, false);
        phi.reinit(cell);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& u_n                      = phi_old.get_value(q);
          const auto& grad_u_n                 = phi_old.get_gradient(q);
          const auto& u_n_gamma                = phi_int.get_value(q);
          const auto& grad_u_n_gamma           = phi_int.get_gradient(q);
          const auto& tensor_product_u_n       = outer_product(u_n, u_n);
          const auto& tensor_product_u_n_gamma = outer_product(u_n_gamma, u_n_gamma);
          const auto& p_n                      = phi_old_press.get_value(q);
          auto p_n_times_identity              = tensor_product_u_n;
          p_n_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d)
            p_n_times_identity[d][d] = p_n;
          phi.submit_value(1.0/((1.0 - gamma)*dt)*u_n_gamma, q);
          phi.submit_gradient(a32*tensor_product_u_n_gamma + a31*tensor_product_u_n -
                              a32/Re*grad_u_n_gamma - a31/Re*grad_u_n + p_n_times_identity, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
  }


  // Assemble rhs face term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  assemble_rhs_face_term_velocity(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi_p(data, true, 0), phi_m(data, false, 0),
                                                                       phi_old_p(data, true, 0), phi_old_m(data, false, 0),
                                                                       phi_old_extr_p(data, true, 0), phi_old_extr_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_v, 1, Number>   phi_old_press_p(data, true, 1), phi_old_press_m(data, false, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_old_p.reinit(face);
        phi_old_p.gather_evaluate(src[0], true, true);
        phi_old_m.reinit(face);
        phi_old_m.gather_evaluate(src[0], true, true);
        phi_old_extr_p.reinit(face);
        phi_old_extr_p.gather_evaluate(src[1], true, false);
        phi_old_extr_m.reinit(face);
        phi_old_extr_m.gather_evaluate(src[1], true, false);
        phi_old_press_p.reinit(face);
        phi_old_press_p.gather_evaluate(src[2], true, false);
        phi_old_press_m.reinit(face);
        phi_old_press_m.gather_evaluate(src[2], true, false);
        phi_p.reinit(face);
        phi_m.reinit(face);
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus                 = phi_p.get_normal_vector(q);
          const auto& avg_grad_u_old         = 0.5*(phi_old_p.get_gradient(q) + phi_old_m.get_gradient(q));
          const auto& avg_tensor_product_u_n = 0.5*(outer_product(phi_old_p.get_value(q), phi_old_extr_p.get_value(q)) +
                                                    outer_product(phi_old_m.get_value(q), phi_old_extr_m.get_value(q)));
          const auto& avg_p_old               = 0.5*(phi_old_press_p.get_value(q) + phi_old_press_m.get_value(q));
          phi_p.submit_value((a21/Re*avg_grad_u_old - a21*avg_tensor_product_u_n)*n_plus - avg_p_old*n_plus, q);
          phi_m.submit_value(-(a21/Re*avg_grad_u_old - a21*avg_tensor_product_u_n)*n_plus + avg_p_old*n_plus, q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi_p(data, true, 0), phi_m(data, false, 0),
                                                                       phi_old_p(data, true, 0), phi_old_m(data, false, 0),
                                                                       phi_int_p(data, true, 0), phi_int_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_v, 1, Number>   phi_old_press_p(data, true, 1), phi_old_press_m(data, false, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++ face) {
        phi_old_p.reinit(face);
        phi_old_p.gather_evaluate(src[0], true, true);
        phi_old_m.reinit(face);
        phi_old_m.gather_evaluate(src[0], true, true);
        phi_int_p.reinit(face);
        phi_int_p.gather_evaluate(src[1], true, true);
        phi_int_m.reinit(face);
        phi_int_m.gather_evaluate(src[1], true, true);
        phi_old_press_p.reinit(face);
        phi_old_press_p.gather_evaluate(src[2], true, false);
        phi_old_press_m.reinit(face);
        phi_old_press_m.gather_evaluate(src[2], true, false);
        phi_p.reinit(face);
        phi_m.reinit(face);
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus                       = phi_p.get_normal_vector(q);
          const auto& avg_grad_u_old               = 0.5*(phi_old_p.get_gradient(q) + phi_old_m.get_gradient(q));
          const auto& avg_grad_u_int               = 0.5*(phi_int_p.get_gradient(q) + phi_int_m.get_gradient(q));
          const auto& avg_tensor_product_u_n       = 0.5*(outer_product(phi_old_p.get_value(q), phi_old_p.get_value(q)) +
                                                          outer_product(phi_old_m.get_value(q), phi_old_m.get_value(q)));
          const auto& avg_tensor_product_u_n_gamma = 0.5*(outer_product(phi_int_p.get_value(q), phi_int_p.get_value(q)) +
                                                          outer_product(phi_int_m.get_value(q), phi_int_m.get_value(q)));
          const auto& avg_p_old                    = 0.5*(phi_old_press_p.get_value(q) + phi_old_press_m.get_value(q));
          phi_p.submit_value((a31/Re*avg_grad_u_old + a32/Re*avg_grad_u_int -
                              a31*avg_tensor_product_u_n - a32*avg_tensor_product_u_n_gamma)*n_plus - avg_p_old*n_plus, q);
          phi_m.submit_value(-(a31/Re*avg_grad_u_old + a32/Re*avg_grad_u_int -
                               a31*avg_tensor_product_u_n - a32*avg_tensor_product_u_n_gamma)*n_plus + avg_p_old*n_plus, q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
  }


  // Assemble rhs boundary term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  assemble_rhs_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                      Vec&                                         dst,
                                      const std::vector<Vec>&                      src,
                                      const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, true, 0),
                                                                       phi_old(data, true, 0),
                                                                       phi_old_extr(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_v, 1, Number>   phi_old_press(data, true, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_old.reinit(face);
        phi_old.gather_evaluate(src[0], true, true);
        phi_old_extr.reinit(face);
        phi_old_extr.gather_evaluate(src[1], true, false);
        phi_old_press.reinit(face);
        phi_old_press.gather_evaluate(src[2], true, false);
        phi.reinit(face);
        const auto boundary_id = data.get_boundary_id(face);
        const auto coef_jump   = (boundary_id == 2) ? 0.0 : C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);
        const double aux_coeff = (boundary_id == 2) ? 0.0 : 1.0;
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus             = phi.get_normal_vector(q);
          const auto& grad_u_old         = phi_old.get_gradient(q);
          const auto& tensor_product_u_n = outer_product(phi_old.get_value(q), phi_old_extr.get_value(q));
          const auto& p_old              = phi_old_press.get_value(q);
          const auto& point_vectorized   = phi.quadrature_point(q);
          auto u_int_m                   = Tensor<1, dim, VectorizedArray<Number>>();
          if(boundary_id == 1) {
            for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
              Point<dim> point;
              for(unsigned int d = 0; d < dim; ++d)
                point[d] = point_vectorized[d][v];
              for(unsigned int d = 0; d < dim; ++d)
                u_int_m[d][v] = vel_boundary.value(point, d);
            }
          }
          const auto tensor_product_u_int_m = outer_product(u_int_m, phi_old_extr.get_value(q));
          const auto lambda                 = (boundary_id == 2) ? 0.0 : std::abs(scalar_product(phi_old_extr.get_value(q), n_plus));
          phi.submit_value((a21/Re*grad_u_old - a21*tensor_product_u_n)*n_plus - p_old*n_plus +
                           a22/Re*2.0*coef_jump*u_int_m -
                           aux_coeff*a22*tensor_product_u_int_m*n_plus + a22*lambda*u_int_m, q);
          phi.submit_normal_derivative(-aux_coeff*theta_v*a22/Re*u_int_m, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, true, 0),
                                                                       phi_old(data, true, 0),
                                                                       phi_int(data, true, 0),
                                                                       phi_int_extr(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_v, 1, Number>   phi_old_press(data, true, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++ face) {
        phi_old.reinit(face);
        phi_old.gather_evaluate(src[0], true, true);
        phi_int.reinit(face);
        phi_int.gather_evaluate(src[1], true, true);
        phi_old_press.reinit(face);
        phi_old_press.gather_evaluate(src[2], true, false);
        phi_int_extr.reinit(face);
        phi_int_extr.gather_evaluate(src[3], true, false);
        phi.reinit(face);
        const auto boundary_id = data.get_boundary_id(face);
        const auto coef_jump   = (boundary_id == 2) ? 0.0 : C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);
        const double aux_coeff = (boundary_id == 2) ? 0.0 : 1.0;
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus                   = phi.get_normal_vector(q);
          const auto& grad_u_old               = phi_old.get_gradient(q);
          const auto& grad_u_int               = phi_int.get_gradient(q);
          const auto& tensor_product_u_n       = outer_product(phi_old.get_value(q), phi_old.get_value(q));
          const auto& tensor_product_u_n_gamma = outer_product(phi_int.get_value(q), phi_int.get_value(q));
          const auto& p_old                    = phi_old_press.get_value(q);
          const auto& point_vectorized         = phi.quadrature_point(q);
          auto u_m                             = Tensor<1, dim, VectorizedArray<Number>>();
          if(boundary_id == 1) {
            for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
              Point<dim> point;
              for(unsigned int d = 0; d < dim; ++d)
                point[d] = point_vectorized[d][v];
              for(unsigned int d = 0; d < dim; ++d)
                u_m[d][v] = vel_boundary.value(point, d);
            }
          }
          const auto tensor_product_u_m = outer_product(u_m, phi_int_extr.get_value(q));
          const auto lambda             = (boundary_id == 2) ? 0.0 : std::abs(scalar_product(phi_int_extr.get_value(q), n_plus));
          phi.submit_value((a31/Re*grad_u_old + a32/Re*grad_u_int -
                           a31*tensor_product_u_n - a32*tensor_product_u_n_gamma)*n_plus - p_old*n_plus +
                           a33/Re*2.0*coef_jump*u_m -
                           aux_coeff*a33*tensor_product_u_m*n_plus + a33*lambda*u_m, q);
          phi.submit_normal_derivative(-aux_coeff*theta_v*a33/Re*u_m, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
  }


  // Put together all the previous steps for velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  vmult_rhs_velocity(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d)
      src[d].update_ghost_values();
    this->data->loop(&NavierStokesProjectionOperator::assemble_rhs_cell_term_velocity,
                     &NavierStokesProjectionOperator::assemble_rhs_face_term_velocity,
                     &NavierStokesProjectionOperator::assemble_rhs_boundary_term_velocity,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble rhs cell term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  assemble_rhs_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number>   phi(data, 1, 1), phi_old(data, 1, 1);
    FEEvaluation<dim, fe_degree_v, n_q_points_1d_p, dim, Number> phi_proj(data, 0, 1);

    const double coeff = (TR_BDF2_stage == 1) ? 300*300*gamma*dt*gamma*dt : 300*300*(1.0 - gamma)*dt*(1.0 - gamma)*dt;

    const double coeff_2 = (TR_BDF2_stage == 1) ? gamma*dt : (1.0 - gamma)*dt;

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_proj.reinit(cell);
      phi_proj.gather_evaluate(src[0], true, false);
      phi_old.reinit(cell);
      phi_old.gather_evaluate(src[1], true, false);
      phi.reinit(cell);
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& u_star_star = phi_proj.get_value(q);
        const auto& p_old       = phi_old.get_value(q);
        phi.submit_value(1.0/coeff*p_old, q);
        phi.submit_gradient(1.0/coeff_2*u_star_star, q);
      }
      phi.integrate_scatter(true, true, dst);
    }
  }


  // Assemble rhs face term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  assemble_rhs_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number>   phi_p(data, true, 1, 1), phi_m(data, false, 1, 1);
    FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_p, dim, Number> phi_proj_p(data, true, 0, 1), phi_proj_m(data, false, 0, 1);

    const double coeff = (TR_BDF2_stage == 1) ? 1.0/(gamma*dt) : 1.0/((1.0 - gamma)*dt);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_proj_p.reinit(face);
      phi_proj_p.gather_evaluate(src[0], true, false);
      phi_proj_m.reinit(face);
      phi_proj_m.gather_evaluate(src[0], true, false);
      phi_p.reinit(face);
      phi_m.reinit(face);
      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus           = phi_p.get_normal_vector(q);
        const auto& avg_u_star_star  = 0.5*(phi_proj_p.get_value(q) + phi_proj_m.get_value(q));
        phi_p.submit_value(-coeff*scalar_product(avg_u_star_star, n_plus), q);
        phi_m.submit_value(coeff*scalar_product(avg_u_star_star, n_plus), q);
      }
      phi_p.integrate_scatter(true, false, dst);
      phi_m.integrate_scatter(true, false, dst);
    }
  }


  // Assemble rhs boundary term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  assemble_rhs_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                      Vec&                                         dst,
                                      const std::vector<Vec>&                      src,
                                      const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number>   phi(data, true, 1, 1);
    FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_p, dim, Number> phi_proj(data, true, 0, 1);

    const double coeff = (TR_BDF2_stage == 1) ? 1.0/(gamma*dt) : 1.0/((1.0 - gamma)*dt);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_proj.reinit(face);
      phi_proj.gather_evaluate(src[0], true, false);
      phi.reinit(face);
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& n_plus = phi.get_normal_vector(q);
        phi.submit_value(-coeff*scalar_product(phi_proj.get_value(q), n_plus), q);
      }
      phi.integrate_scatter(true, false, dst);
    }
  }


  // Put together all the previous steps for pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  vmult_rhs_pressure(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d)
      src[d].update_ghost_values();
    this->data->loop(&NavierStokesProjectionOperator::assemble_rhs_cell_term_pressure,
                     &NavierStokesProjectionOperator::assemble_rhs_face_term_pressure,
                     &NavierStokesProjectionOperator::assemble_rhs_boundary_term_pressure,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble cell term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  assemble_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(TR_BDF2_stage == 1) {
      FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0), phi_old_extr(data, 0);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi.reinit(cell);
        phi.gather_evaluate(src, true, true);
        phi_old_extr.reinit(cell);
        phi_old_extr.gather_evaluate(u_extr, true, false);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& u_int                = phi.get_value(q);
          const auto& grad_u_int           = phi.get_gradient(q);
          const auto& u_n_gamma_ov_2       = phi_old_extr.get_value(q);
          const auto& tensor_product_u_int = outer_product(u_int, u_n_gamma_ov_2);
          phi.submit_value(1.0/(gamma*dt)*u_int, q);
          phi.submit_gradient(-a22*tensor_product_u_int + a22/Re*grad_u_int, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0), phi_int_extr(data, 0);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi.reinit(cell);
        phi.gather_evaluate(src, true, true);
        phi_int_extr.reinit(cell);
        phi_int_extr.gather_evaluate(u_extr, true, false);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& u_curr                   = phi.get_value(q);
          const auto& grad_u_curr              = phi.get_gradient(q);
          const auto& u_n1_int                 = phi_int_extr.get_value(q);
          const auto& tensor_product_u_curr    = outer_product(u_curr, u_n1_int);
          phi.submit_value(1.0/((1.0 - gamma)*dt)*u_curr, q);
          phi.submit_gradient(-a33*tensor_product_u_curr + a33/Re*grad_u_curr, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
  }


  // Assemble face term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  assemble_face_term_velocity(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi_p(data, true, 0), phi_m(data, false, 0),
                                                                       phi_old_extr_p(data, true, 0), phi_old_extr_m(data, false, 0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_p.reinit(face);
        phi_p.gather_evaluate(src, true, true);
        phi_m.reinit(face);
        phi_m.gather_evaluate(src, true, true);
        const auto coef_jump = C_u*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                        std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));
        phi_old_extr_p.reinit(face);
        phi_old_extr_p.gather_evaluate(u_extr, true, false);
        phi_old_extr_m.reinit(face);
        phi_old_extr_m.gather_evaluate(u_extr, true, false);
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus                   = phi_p.get_normal_vector(q);
          const auto& avg_grad_u_int           = 0.5*(phi_p.get_gradient(q) + phi_m.get_gradient(q));
          const auto& jump_u_int               = phi_p.get_value(q) - phi_m.get_value(q);
          const auto& avg_tensor_product_u_int = 0.5*(outer_product(phi_p.get_value(q), phi_old_extr_p.get_value(q)) +
                                                      outer_product(phi_m.get_value(q), phi_old_extr_m.get_value(q)));
          const auto  lambda                   = std::max(std::abs(scalar_product(phi_old_extr_p.get_value(q), n_plus)),
                                                          std::abs(scalar_product(phi_old_extr_m.get_value(q), n_plus)));
          phi_p.submit_value(a22/Re*(-avg_grad_u_int*n_plus + coef_jump*jump_u_int) +
                             a22*avg_tensor_product_u_int*n_plus + 0.5*a22*lambda*jump_u_int, q);
          phi_m.submit_value(-a22/Re*(-avg_grad_u_int*n_plus + coef_jump*jump_u_int) -
                              a22*avg_tensor_product_u_int*n_plus - 0.5*a22*lambda*jump_u_int, q);
          phi_p.submit_normal_derivative(-theta_v*a22/Re*0.5*jump_u_int, q);
          phi_m.submit_normal_derivative(-theta_v*a22/Re*0.5*jump_u_int, q);
        }
        phi_p.integrate_scatter(true, true, dst);
        phi_m.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi_p(data, true, 0), phi_m(data, false, 0),
                                                                       phi_extr_p(data, true, 0), phi_extr_m(data, false, 0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_p.reinit(face);
        phi_p.gather_evaluate(src, true, true);
        phi_m.reinit(face);
        phi_m.gather_evaluate(src, true, true);
        phi_extr_p.reinit(face);
        phi_extr_p.gather_evaluate(u_extr, true, false);
        phi_extr_m.reinit(face);
        phi_extr_m.gather_evaluate(u_extr, true, false);
        const auto coef_jump = C_u*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                        std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus               = phi_p.get_normal_vector(q);
          const auto& avg_grad_u           = 0.5*(phi_p.get_gradient(q) + phi_m.get_gradient(q));
          const auto& jump_u               = phi_p.get_value(q) - phi_m.get_value(q);
          const auto& avg_tensor_product_u = 0.5*(outer_product(phi_p.get_value(q), phi_extr_p.get_value(q)) +
                                                  outer_product(phi_m.get_value(q), phi_extr_m.get_value(q)));
          const auto  lambda               = std::max(std::abs(scalar_product(phi_extr_p.get_value(q), n_plus)),
                                                      std::abs(scalar_product(phi_extr_m.get_value(q), n_plus)));
          phi_p.submit_value(a33/Re*(-avg_grad_u*n_plus + coef_jump*jump_u) +
                             a33*avg_tensor_product_u*n_plus + 0.5*a33*lambda*jump_u, q);
          phi_m.submit_value(-a33/Re*(-avg_grad_u*n_plus + coef_jump*jump_u) -
                              a33*avg_tensor_product_u*n_plus - 0.5*a33*lambda*jump_u, q);
          phi_p.submit_normal_derivative(-theta_v*a33/Re*0.5*jump_u, q);
          phi_m.submit_normal_derivative(-theta_v*a33/Re*0.5*jump_u, q);
        }
        phi_p.integrate_scatter(true, true, dst);
        phi_m.integrate_scatter(true, true, dst);
      }
    }
  }


  // Assemble boundary term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  assemble_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const Vec&                                   src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, true, 0), phi_old_extr(data, true, 0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi.reinit(face);
        phi.gather_evaluate(src, true, true);
        const auto boundary_id  = data.get_boundary_id(face);
        const auto coef_jump    = (boundary_id == 2) ? 0.0 : C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);
        const double coef_trasp = (boundary_id == 2) ? 1.0 : 0.0;
        const double aux_coeff  = (boundary_id == 2) ? 0.0 : 1.0;
        phi_old_extr.reinit(face);
        phi_old_extr.gather_evaluate(u_extr, true, false);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus               = phi.get_normal_vector(q);
          const auto& grad_u_int           = phi.get_gradient(q);
          const auto& u_int                = phi.get_value(q);
          const auto& tensor_product_u_int = outer_product(phi.get_value(q), phi_old_extr.get_value(q));
          const auto  lambda               = (boundary_id == 2) ? 0.0 : std::abs(scalar_product(phi_old_extr.get_value(q), n_plus));
          phi.submit_value(a22/Re*(-aux_coeff*grad_u_int*n_plus + 2.0*coef_jump*u_int) +
                           a22*coef_trasp*tensor_product_u_int*n_plus + a22*lambda*u_int, q);
          phi.submit_normal_derivative(-aux_coeff*theta_v*a22/Re*u_int, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, true, 0), phi_extr(data, true, 0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi.reinit(face);
        phi.gather_evaluate(src, true, true);
        const auto boundary_id  = data.get_boundary_id(face);
        const auto coef_jump    = (boundary_id == 2) ? 0.0 : C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);
        const double coef_trasp = (boundary_id == 2) ? 1.0 : 0.0;
        const double aux_coeff  = (boundary_id == 2) ? 0.0 : 1.0;
        phi_extr.reinit(face);
        phi_extr.gather_evaluate(u_extr, true, false);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus           = phi.get_normal_vector(q);
          const auto& grad_u           = phi.get_gradient(q);
          const auto& u                = phi.get_value(q);
          const auto& tensor_product_u = outer_product(phi.get_value(q), phi_extr.get_value(q));
          const auto  lambda           = (boundary_id == 2) ? 0.0 : std::abs(scalar_product(phi_extr.get_value(q), n_plus));
          phi.submit_value(a33/Re*(-aux_coeff*grad_u*n_plus + 2.0*coef_jump*u) +
                           a33*coef_trasp*tensor_product_u*n_plus + a33*lambda*u, q);
          phi.submit_normal_derivative(-aux_coeff*theta_v*a33/Re*u, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
  }


  // Assemble cell term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  assemble_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number> phi(data, 1, 1);

    const double coeff = (TR_BDF2_stage == 1) ? 300*300*gamma*dt*gamma*dt : 300*300*(1.0 - gamma)*dt*(1.0 - gamma)*dt;

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, true, true);
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        phi.submit_gradient(phi.get_gradient(q), q);
        phi.submit_value(1.0/coeff*phi.get_value(q), q);
      }
      phi.integrate_scatter(true, true, dst);
    }
  }


  // Assemble face term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  assemble_face_term_pressure(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number> phi_p(data, true, 1, 1), phi_m(data, false, 1, 1);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_p.reinit(face);
      phi_p.gather_evaluate(src, true, true);
      phi_m.reinit(face);
      phi_m.gather_evaluate(src, true, true);
      const auto coef_jump = C_p*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                      std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));
      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus        = phi_p.get_normal_vector(q);
        const auto& avg_grad_pres = 0.5*(phi_p.get_gradient(q) + phi_m.get_gradient(q));
        const auto& jump_pres     = phi_p.get_value(q) - phi_m.get_value(q);
        phi_p.submit_value(-scalar_product(avg_grad_pres, n_plus) + coef_jump*jump_pres, q);
        phi_m.submit_value(scalar_product(avg_grad_pres, n_plus) - coef_jump*jump_pres, q);
        phi_p.submit_gradient(-theta_p*0.5*jump_pres*n_plus, q);
        phi_m.submit_gradient(-theta_p*0.5*jump_pres*n_plus, q);
      }
      phi_p.integrate_scatter(true, true, dst);
      phi_m.integrate_scatter(true, true, dst);
    }
  }


  // Put together all previous steps
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  apply_add(Vec& dst, const Vec& src) const {
    if(NS_stage == 1) {
      this->data->loop(&NavierStokesProjectionOperator::assemble_cell_term_velocity,
                       &NavierStokesProjectionOperator::assemble_face_term_velocity,
                       &NavierStokesProjectionOperator::assemble_boundary_term_velocity,
                       this, dst, src, false,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
    }
    else if(NS_stage == 2) {
      this->data->loop(&NavierStokesProjectionOperator::assemble_cell_term_pressure,
                       &NavierStokesProjectionOperator::assemble_face_term_pressure,
                       &NavierStokesProjectionOperator::assemble_boundary_term_pressure,
                       this, dst, src, false,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
    }
    else if(NS_stage == 3) {
      this->data->cell_loop(&NavierStokesProjectionOperator::assemble_cell_term_projection_grad_p,
                            this, dst, src, false);
    }
    else
      Assert(false, ExcNotImplemented());
  }


  // Assemble cell term for the projection of gradient of pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  assemble_cell_term_projection_grad_p(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const Vec&                                   src,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, true, false);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(phi.get_value(q), q);
      phi.integrate_scatter(true, false, dst);
    }
  }


  // Assemble rhs cell term for the projection of gradient of pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  assemble_rhs_cell_term_projection_grad_p(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const Vec&                                   src,
                                           const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0);
    FEEvaluation<dim, fe_degree_p, n_q_points_1d_v, 1, Number>   phi_pres(data, 1);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_pres.reinit(cell);
      phi_pres.gather_evaluate(src, false, true);
      phi.reinit(cell);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(phi_pres.get_gradient(q), q);
      phi.integrate_scatter(true, false, dst);
    }
  }


  // Put together all the previous steps for pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  vmult_grad_p_projection(Vec& dst, const Vec& src) const {
    this->data->cell_loop(&NavierStokesProjectionOperator::assemble_rhs_cell_term_projection_grad_p,
                          this, dst, src, true);
  }


  // Assemble diagonal cell term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  assemble_diagonal_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const Vec&                                   src,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(TR_BDF2_stage == 1) {
      FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0), phi_old_extr(data, 0);

      AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);
      Tensor<1, dim, VectorizedArray<Number>> tmp;
      for(unsigned int d = 0; d < dim; ++d)
        tmp[d] = make_vectorized_array<Number>(1.0);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_old_extr.reinit(cell);
        phi_old_extr.gather_evaluate(u_extr, true, false);
        phi.reinit(cell);
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
            phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
          phi.submit_dof_value(tmp, i);
          phi.evaluate(true, true);
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& u_int                = phi.get_value(q);
            const auto& grad_u_int           = phi.get_gradient(q);
            const auto& u_n_gamma_ov_2       = phi_old_extr.get_value(q);
            const auto& tensor_product_u_int = outer_product(u_int, u_n_gamma_ov_2);
            phi.submit_value(1.0/(gamma*dt)*u_int, q);
            phi.submit_gradient(-a22*tensor_product_u_int + a22/Re*grad_u_int, q);
          }
          phi.integrate(true, true);
          diagonal[i] = phi.get_dof_value(i);
        }
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i)
          phi.submit_dof_value(diagonal[i], i);
        phi.distribute_local_to_global(dst);
      }
    }
    else {
      FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0), phi_int_extr(data, 0);

      AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);
      Tensor<1, dim, VectorizedArray<Number>> tmp;
      for(unsigned int d = 0; d < dim; ++d)
        tmp[d] = make_vectorized_array<Number>(1.0);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_int_extr.reinit(cell);
        phi_int_extr.gather_evaluate(u_extr, true, false);
        phi.reinit(cell);
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
            phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
          phi.submit_dof_value(tmp, i);
          phi.evaluate(true, true);
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& u_curr                   = phi.get_value(q);
            const auto& grad_u_curr              = phi.get_gradient(q);
            const auto& u_n1_int                 = phi_int_extr.get_value(q);
            const auto& tensor_product_u_curr    = outer_product(u_curr, u_n1_int);
            phi.submit_value(1.0/((1.0 - gamma)*dt)*u_curr, q);
            phi.submit_gradient(-a33*tensor_product_u_curr + a33/Re*grad_u_curr, q);
          }
          phi.integrate(true, true);
          diagonal[i] = phi.get_dof_value(i);
        }
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i)
          phi.submit_dof_value(diagonal[i], i);
        phi.distribute_local_to_global(dst);
      }
    }
  }


  // Assemble diagonal face term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  assemble_diagonal_face_term_velocity(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const Vec&                                   src,
                                       const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi_p(data, true, 0), phi_m(data, false, 0),
                                                                       phi_old_extr_p(data, true, 0), phi_old_extr_m(data, false, 0);

      AssertDimension(phi_p.dofs_per_component, phi_m.dofs_per_component);
      AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal_p(phi_p.dofs_per_component), diagonal_m(phi_m.dofs_per_component);
      Tensor<1, dim, VectorizedArray<Number>> tmp;
      for(unsigned int d = 0; d < dim; ++d)
        tmp[d] = make_vectorized_array<Number>(1.0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_old_extr_p.reinit(face);
        phi_old_extr_p.gather_evaluate(u_extr, true, false);
        phi_old_extr_m.reinit(face);
        phi_old_extr_m.gather_evaluate(u_extr, true, false);
        phi_p.reinit(face);
        phi_m.reinit(face);
        const auto coef_jump = C_u*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                        std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));
        for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi_p.dofs_per_component; ++j) {
            phi_p.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
            phi_m.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
          }
          phi_p.submit_dof_value(tmp, i);
          phi_p.evaluate(true, true);
          phi_m.submit_dof_value(tmp, i);
          phi_m.evaluate(true, true);
          for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
            const auto& n_plus                   = phi_p.get_normal_vector(q);
            const auto& avg_grad_u_int           = 0.5*(phi_p.get_gradient(q) + phi_m.get_gradient(q));
            const auto& jump_u_int               = phi_p.get_value(q) - phi_m.get_value(q);
            const auto& avg_tensor_product_u_int = 0.5*(outer_product(phi_p.get_value(q), phi_old_extr_p.get_value(q)) +
                                                        outer_product(phi_m.get_value(q), phi_old_extr_m.get_value(q)));
            const auto  lambda                   = std::max(std::abs(scalar_product(phi_old_extr_p.get_value(q), n_plus)),
                                                            std::abs(scalar_product(phi_old_extr_m.get_value(q), n_plus)));
            phi_p.submit_value(a22/Re*(-avg_grad_u_int*n_plus + coef_jump*jump_u_int) +
                               a22*avg_tensor_product_u_int*n_plus + 0.5*a22*lambda*jump_u_int , q);
            phi_m.submit_value(-a22/Re*(-avg_grad_u_int*n_plus + coef_jump*jump_u_int) -
                               a22*avg_tensor_product_u_int*n_plus - 0.5*a22*lambda*jump_u_int, q);
            phi_p.submit_normal_derivative(-theta_v*0.5*a22/Re*jump_u_int, q);
            phi_m.submit_normal_derivative(-theta_v*0.5*a22/Re*jump_u_int, q);
          }
          phi_p.integrate(true, true);
          diagonal_p[i] = phi_p.get_dof_value(i);
          phi_m.integrate(true, true);
          diagonal_m[i] = phi_m.get_dof_value(i);
        }
        for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
          phi_p.submit_dof_value(diagonal_p[i], i);
          phi_m.submit_dof_value(diagonal_m[i], i);
        }
        phi_p.distribute_local_to_global(dst);
        phi_m.distribute_local_to_global(dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi_p(data, true, 0), phi_m(data, false, 0),
                                                                       phi_extr_p(data, true, 0), phi_extr_m(data, false, 0);

      AssertDimension(phi_p.dofs_per_component, phi_m.dofs_per_component);
      AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal_p(phi_p.dofs_per_component), diagonal_m(phi_m.dofs_per_component);
      Tensor<1, dim, VectorizedArray<Number>> tmp;
      for(unsigned int d = 0; d < dim; ++d)
        tmp[d] = make_vectorized_array<Number>(1.0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_extr_p.reinit(face);
        phi_extr_p.gather_evaluate(u_extr, true, false);
        phi_extr_m.reinit(face);
        phi_extr_m.gather_evaluate(u_extr, true, false);
        phi_p.reinit(face);
        phi_m.reinit(face);
        const auto coef_jump = C_u*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                        std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));
        for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi_p.dofs_per_component; ++j) {
            phi_p.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
            phi_m.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
          }
          phi_p.submit_dof_value(tmp, i);
          phi_p.evaluate(true, true);
          phi_m.submit_dof_value(tmp, i);
          phi_m.evaluate(true, true);
          for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
            const auto& n_plus               = phi_p.get_normal_vector(q);
            const auto& avg_grad_u           = 0.5*(phi_p.get_gradient(q) + phi_m.get_gradient(q));
            const auto& jump_u               = phi_p.get_value(q) - phi_m.get_value(q);
            const auto& avg_tensor_product_u = 0.5*(outer_product(phi_p.get_value(q), phi_extr_p.get_value(q)) +
                                                    outer_product(phi_m.get_value(q), phi_extr_m.get_value(q)));
            const auto  lambda               = std::max(std::abs(scalar_product(phi_extr_p.get_value(q), n_plus)),
                                                        std::abs(scalar_product(phi_extr_m.get_value(q), n_plus)));
            phi_p.submit_value(a33/Re*(-avg_grad_u*n_plus + coef_jump*jump_u) +
                               a33*avg_tensor_product_u*n_plus + 0.5*a33*lambda*jump_u, q);
            phi_m.submit_value(-a33/Re*(-avg_grad_u*n_plus + coef_jump*jump_u) -
                               a33*avg_tensor_product_u*n_plus - 0.5*a33*lambda*jump_u, q);
            phi_p.submit_normal_derivative(-theta_v*0.5*a33/Re*jump_u, q);
            phi_m.submit_normal_derivative(-theta_v*0.5*a33/Re*jump_u, q);
          }
          phi_p.integrate(true, true);
          diagonal_p[i] = phi_p.get_dof_value(i);
          phi_m.integrate(true, true);
          diagonal_m[i] = phi_m.get_dof_value(i);
        }
        for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
          phi_p.submit_dof_value(diagonal_p[i], i);
          phi_m.submit_dof_value(diagonal_m[i], i);
        }
        phi_p.distribute_local_to_global(dst);
        phi_m.distribute_local_to_global(dst);
      }
    }
  }


  // Assemble boundary term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  assemble_diagonal_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const Vec&                                   src,
                                           const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, true, 0), phi_old_extr(data, true, 0);

      AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);
      Tensor<1, dim, VectorizedArray<Number>> tmp;
      for(unsigned int d = 0; d < dim; ++d)
        tmp[d] = make_vectorized_array<Number>(1.0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_old_extr.reinit(face);
        phi_old_extr.gather_evaluate(u_extr, true, false);
        phi.reinit(face);
        const auto boundary_id  = data.get_boundary_id(face);
        const auto coef_jump    = (boundary_id == 2) ? 0.0 : C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);
        const double coef_trasp = (boundary_id == 2) ? 1.0 : 0.0;
        const double aux_coeff  = (boundary_id == 2) ? 0.0 : 1.0;
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
            phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
          phi.submit_dof_value(tmp, i);
          phi.evaluate(true, true);
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& n_plus               = phi.get_normal_vector(q);
            const auto& grad_u_int           = phi.get_gradient(q);
            const auto& u_int                = phi.get_value(q);
            const auto& tensor_product_u_int = outer_product(phi.get_value(q), phi_old_extr.get_value(q));
            const auto  lambda               = (boundary_id == 2) ? 0.0 : std::abs(scalar_product(phi_old_extr.get_value(q), n_plus));
            phi.submit_value(a22/Re*(-aux_coeff*grad_u_int*n_plus + 2.0*coef_jump*u_int) +
                             a22*coef_trasp*tensor_product_u_int*n_plus + a22*lambda*u_int, q);
            phi.submit_normal_derivative(-aux_coeff*theta_v*a22/Re*u_int, q);
          }
          phi.integrate(true, true);
          diagonal[i] = phi.get_dof_value(i);
        }
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i)
          phi.submit_dof_value(diagonal[i], i);
        phi.distribute_local_to_global(dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, true, 0), phi_extr(data, true, 0);

      AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);
      Tensor<1, dim, VectorizedArray<Number>> tmp;
      for(unsigned int d = 0; d < dim; ++d)
        tmp[d] = make_vectorized_array<Number>(1.0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_extr.reinit(face);
        phi_extr.gather_evaluate(u_extr, true, false);
        phi.reinit(face);
        const auto boundary_id  = data.get_boundary_id(face);
        const auto coef_jump    = (boundary_id == 2) ? 0.0 : C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);
        const double coef_trasp = (boundary_id == 2) ? 1.0 : 0.0;
        const double aux_coeff  = (boundary_id == 2) ? 0.0 : 1.0;
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
            phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
          phi.submit_dof_value(tmp, i);
          phi.evaluate(true, true);
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& n_plus           = phi.get_normal_vector(q);
            const auto& grad_u           = phi.get_gradient(q);
            const auto& u                = phi.get_value(q);
            const auto& tensor_product_u = outer_product(phi.get_value(q), phi_extr.get_value(q));
            const auto  lambda           = (boundary_id == 2) ? 0.0 : std::abs(scalar_product(phi_extr.get_value(q), n_plus));
            phi.submit_value(a33/Re*(-aux_coeff*grad_u*n_plus + coef_jump*u) +
                             a33*coef_trasp*tensor_product_u*n_plus + a33*lambda*u, q);
            phi.submit_normal_derivative(-aux_coeff*theta_v*a33/Re*u, q);
          }
          phi.integrate(true, true);
          diagonal[i] = phi.get_dof_value(i);
        }
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i)
          phi.submit_dof_value(diagonal[i], i);
        phi.distribute_local_to_global(dst);
      }
    }
  }


  // Assemble diagonal cell term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  assemble_diagonal_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const Vec&                                   src,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number> phi(data, 1, 1);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    const double coeff = (TR_BDF2_stage == 1) ? 300*300*gamma*dt*gamma*dt : 300*300*(1.0 - gamma)*dt*(1.0 - gamma)*dt;

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
          phi.submit_dof_value(VectorizedArray<Number>(), j);
        phi.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        phi.evaluate(true, true);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          phi.submit_value(1.0/coeff*phi.get_value(q), q);
          phi.submit_gradient(phi.get_gradient(q), q);
        }
        phi.integrate(true, true);
        diagonal[i] = phi.get_dof_value(i);
      }
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i)
        phi.submit_dof_value(diagonal[i], i);
      phi.distribute_local_to_global(dst);
    }
  }


  // Assemble diagonal face term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::
  assemble_diagonal_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const Vec&                                   src,
                                       const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number> phi_p(data, true, 1, 1), phi_m(data, false, 1, 1);

    AssertDimension(phi_p.dofs_per_component, phi_m.dofs_per_component);
    AlignedVector<VectorizedArray<Number>> diagonal_p(phi_p.dofs_per_component),
                                           diagonal_m(phi_m.dofs_per_component);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_p.reinit(face);
      phi_m.reinit(face);
      const auto coef_jump = C_p*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                      std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));
      for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi_p.dofs_per_component; ++j) {
          phi_p.submit_dof_value(VectorizedArray<Number>(), j);
          phi_m.submit_dof_value(VectorizedArray<Number>(), j);
        }
        phi_p.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        phi_m.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        phi_p.evaluate(true, true);
        phi_m.evaluate(true, true);
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus        = phi_p.get_normal_vector(q);
          const auto& avg_grad_pres = 0.5*(phi_p.get_gradient(q) + phi_m.get_gradient(q));
          const auto& jump_pres     = phi_p.get_value(q) - phi_m.get_value(q);
          phi_p.submit_value(-scalar_product(avg_grad_pres, n_plus) + coef_jump*jump_pres, q);
          phi_m.submit_value(scalar_product(avg_grad_pres, n_plus) - coef_jump*jump_pres, q);
          phi_p.submit_gradient(-theta_p*0.5*jump_pres*n_plus, q);
          phi_m.submit_gradient(-theta_p*0.5*jump_pres*n_plus, q);
        }
        phi_p.integrate(true, true);
        diagonal_p[i] = phi_p.get_dof_value(i);
        phi_m.integrate(true, true);
        diagonal_m[i] = phi_m.get_dof_value(i);
      }
      for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
        phi_p.submit_dof_value(diagonal_p[i], i);
        phi_m.submit_dof_value(diagonal_m[i], i);
      }
      phi_p.distribute_local_to_global(dst);
      phi_m.distribute_local_to_global(dst);
    }
  }


  // Put together all previous steps
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec, Number>::compute_diagonal() {
    Assert(NS_stage == 1 || NS_stage == 2, ExcInternalError());
    if(NS_stage == 1) {
      this->inverse_diagonal_entries.reset(new DiagonalMatrix<Vec>());
      auto& inverse_diagonal = this->inverse_diagonal_entries->get_vector();
      this->data->initialize_dof_vector(inverse_diagonal, 0);
      Vec dummy;
      dummy.reinit(inverse_diagonal.local_size());
      this->data->loop(&NavierStokesProjectionOperator::assemble_diagonal_cell_term_velocity,
                       &NavierStokesProjectionOperator::assemble_diagonal_face_term_velocity,
                       &NavierStokesProjectionOperator::assemble_diagonal_boundary_term_velocity,
                       this, inverse_diagonal, dummy, false,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
      for(unsigned int i = 0; i < inverse_diagonal.local_size(); ++i) {
        Assert(inverse_diagonal.local_element(i) != 0.0,
               ExcMessage("No diagonal entry in a definite operator should be zero"));
        inverse_diagonal.local_element(i) = 1.0/inverse_diagonal.local_element(i);
      }
    }
    else if(NS_stage == 2) {
      this->inverse_diagonal_entries.reset(new DiagonalMatrix<Vec>());
      auto& inverse_diagonal = this->inverse_diagonal_entries->get_vector();
      this->data->initialize_dof_vector(inverse_diagonal, 1);
      Vec dummy;
      dummy.reinit(inverse_diagonal.local_size());
      this->data->loop(&NavierStokesProjectionOperator::assemble_diagonal_cell_term_pressure,
                       &NavierStokesProjectionOperator::assemble_diagonal_face_term_pressure,
                       &NavierStokesProjectionOperator::assemble_diagonal_boundary_term_pressure,
                       this, inverse_diagonal, dummy, false,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
      for(unsigned int i = 0; i < inverse_diagonal.local_size(); ++i) {
        Assert(inverse_diagonal.local_element(i) != 0.0,
               ExcMessage("No diagonal entry in a definite operator should be zero"));
        inverse_diagonal.local_element(i) = 1.0/inverse_diagonal.local_element(i);
      }
    }
  }



  // @sect3{The <code>NavierStokesProjection</code> class}

  // Now for the main class of the program. It implements the various versions
  // of the projection method for Navier-Stokes equations.
  //
  template<int dim>
  class NavierStokesProjection {
  public:
    NavierStokesProjection(RunTimeParameters::Data_Storage& data);

    void run(const bool verbose = false, const unsigned int output_interval = 10);

  protected:
    const double       t_0;
    const double       T;
    const double       gamma;         //--- TR-BDF2 parameter
    unsigned int       TR_BDF2_stage; //--- Flag to check at which current stage of TR-BDF2 are
    const double       Re;
    double             dt;

    EquationData::Pressure<dim> pres_init;

    parallel::distributed::Triangulation<dim> triangulation;

    FESystem<dim> fe_velocity;
    FESystem<dim> fe_pressure;

    DoFHandler<dim> dof_handler_velocity;
    DoFHandler<dim> dof_handler_pressure;

    QGauss<dim> quadrature_pressure;
    QGauss<dim> quadrature_velocity;

    LinearAlgebra::distributed::Vector<double> pres_n;
    LinearAlgebra::distributed::Vector<double> pres_int;
    LinearAlgebra::distributed::Vector<double> rhs_p;

    LinearAlgebra::distributed::Vector<double> u_n;
    LinearAlgebra::distributed::Vector<double> u_n_minus_1;
    LinearAlgebra::distributed::Vector<double> u_extr;
    LinearAlgebra::distributed::Vector<double> u_n_gamma;
    LinearAlgebra::distributed::Vector<double> u_star;
    LinearAlgebra::distributed::Vector<double> u_tmp;
    LinearAlgebra::distributed::Vector<double> rhs_u;

    LinearAlgebra::distributed::Vector<double> u_n_k;
    LinearAlgebra::distributed::Vector<double> u_n_gamma_k;

    Vector<double> Linfty_error_per_cell_vel;

    DeclException2(ExcInvalidTimeStep,
                   double,
                   double,
                   << " The time step " << arg1 << " is out of range."
                   << std::endl
                   << " The permitted range is (0," << arg2 << "]");

    void create_triangulation();

    void setup_dofs();

    void initialize();

    void interpolate_velocity();

    void diffusion_step();

    void projection_step();

    void project_grad(const unsigned int flag);

    double get_maximal_velocity();

    double get_maximal_difference();

    void output_results(const unsigned int step);

    void refine_mesh();

    void interpolate_max_res(const unsigned int level);

    void save_max_res();

  private:
    std::shared_ptr<MatrixFree<dim, double>> matrix_free_storage;

    std::vector<QGauss<1>> quadratures;
    std::vector<const DoFHandler<dim>*> dof_handlers;

    NavierStokesProjectionOperator<dim, EquationData::degree_p, EquationData::degree_p + 1,
                                   EquationData::degree_p + 1, EquationData::degree_p + 2,
                                   LinearAlgebra::distributed::Vector<double>, double> navier_stokes_matrix;

    MGLevelObject<NavierStokesProjectionOperator<dim, EquationData::degree_p, EquationData::degree_p + 1,
                                                 EquationData::degree_p + 1, EquationData::degree_p + 2,
                                                 LinearAlgebra::distributed::Vector<float>, float>> mg_matrices;

    AffineConstraints<double> constraints_velocity, constraints_pressure;

    unsigned int vel_max_its;
    unsigned int vel_Krylov_size;
    unsigned int vel_off_diagonals;
    unsigned int vel_update_prec;
    double       vel_eps;
    double       vel_diag_strength;

    unsigned int max_loc_refinements;
    unsigned int min_loc_refinements;
    unsigned int refinement_iterations;

    std::string saving_dir;

    ConditionalOStream pcout;

    std::ofstream      time_out;
    ConditionalOStream ptime_out;
    TimerOutput        time_table;

    std::ofstream output_n_dofs_velocity;
    std::ofstream output_n_dofs_pressure;
  };


  // @sect4{ <code>NavierStokesProjection::NavierStokesProjection</code> }

  // In the constructor, we just read all the data from the
  // <code>Data_Storage</code> object that is passed as an argument, verify that
  // the data we read are reasonable and, finally, create the triangulation and
  // load the initial data.
  template<int dim>
  NavierStokesProjection<dim>::NavierStokesProjection(RunTimeParameters::Data_Storage& data):
    t_0(data.initial_time),
    T(data.final_time),
    gamma(2.0 - std::sqrt(2.0)),  //--- Save also in the NavierStokes class the TR-BDF2 parameter value
    TR_BDF2_stage(1),             //--- Initialize the flag for the TR_BDF2 stage
    Re(data.Reynolds),
    dt(data.dt),
    pres_init(data.initial_time),
    triangulation(MPI_COMM_WORLD, Triangulation<dim>::limit_level_difference_at_vertices,
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
    fe_velocity(FE_DGQ<dim>(EquationData::degree_p + 1), dim),
    fe_pressure(FE_DGQ<dim>(EquationData::degree_p), 1),
    dof_handler_velocity(triangulation),
    dof_handler_pressure(triangulation),
    quadrature_pressure(EquationData::degree_p + 1),
    quadrature_velocity(EquationData::degree_p + 2),
    navier_stokes_matrix(data),
    vel_max_its(data.vel_max_iterations),
    vel_Krylov_size(data.vel_Krylov_size),
    vel_off_diagonals(data.vel_off_diagonals),
    vel_update_prec(data.vel_update_prec),
    vel_eps(data.vel_eps),
    vel_diag_strength(data.vel_diag_strength),
    max_loc_refinements(data.max_loc_refinements),
    min_loc_refinements(data.min_loc_refinements),
    refinement_iterations(data.refinement_iterations),
    saving_dir(data.dir),
    pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    time_out("./" + data.dir + "/time_analysis_" +
             Utilities::int_to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)) + "proc.dat"),
    ptime_out(time_out, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    time_table(ptime_out, TimerOutput::summary, TimerOutput::cpu_and_wall_times),
    output_n_dofs_velocity("./" + data.dir + "/n_dofs_velocity.dat", std::ofstream::out),
    output_n_dofs_pressure("./" + data.dir + "/n_dofs_pressure.dat", std::ofstream::out) {
      if(EquationData::degree_p < 1) {
        pcout
        << " WARNING: The chosen pair of finite element spaces is not stable."
        << std::endl
        << " The obtained results will be nonsense" << std::endl;
      }

      AssertThrow(!((dt <= 0.0) || (dt > 0.5*T)), ExcInvalidTimeStep(dt, 0.5*T));

      matrix_free_storage = std::make_shared<MatrixFree<dim, double>>();

      constraints_velocity.clear();
      constraints_pressure.clear();

      create_triangulation();
      setup_dofs();
      initialize();
  }


  // @sect4{<code>NavierStokesProjection::create_triangulation_and_dofs</code>}

  // The method that creates the triangulation and refines it the needed number
  // of times.
  //
  template<int dim>
  void NavierStokesProjection<dim>::create_triangulation() {
    TimerOutput::Scope t(time_table, "Create triangulation");

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);
    std::string   filename = "full_0.ucd";
    std::ifstream file(filename);
    Assert(file, ExcFileNotOpen(filename.c_str()));
    grid_in.read_ucd(file);

    std::ifstream inlet_data("INLET_AIR_0.ucd");
    std::string curr_line;
    std::getline(inlet_data, curr_line);
    std::getline(inlet_data, curr_line);
    std::vector<double> xx(315), yy(315), zz(315);
    double index, x, y, z;
    for(unsigned int i = 0; i < 315; ++i) {
      std::getline(inlet_data, curr_line);
      std::istringstream iss(curr_line);
      iss.precision(16);
      iss >> index >> x >> y >> z;
      xx[i] = x;
      yy[i] = y;
      zz[i] = z;
    }
    for(const auto& cell: triangulation.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
          if(cell->face(f)->at_boundary()) {
            bool of_interest = true;
            for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
              unsigned int i = 0;
              while(i < 315) {
                if(std::abs(cell->face(f)->vertex(v)(0) - xx[i]) < 1e-9 &&
                   std::abs(cell->face(f)->vertex(v)(1) - yy[i]) < 1e-9 &&
                   std::abs(cell->face(f)->vertex(v)(2) - zz[i]) < 1e-9)
                   break;
                i++;
              }
              if(i == 315) {
                of_interest = false;
                break;
              }
            }
            if(of_interest)
              cell->face(f)->set_boundary_id(1);
          }
        }
      }
    }

    std::ifstream outlet_data("OUTLET_AIR_0.ucd");
    std::getline(outlet_data, curr_line);
    std::getline(outlet_data, curr_line);
    xx.resize(1035);
    yy.resize(1035);
    zz.resize(1035);
    for(unsigned int i = 0; i < 1035; ++i) {
      std::getline(outlet_data, curr_line);
      std::istringstream iss(curr_line);
      iss.precision(16);
      iss >> index >> x >> y >> z;
      xx[i] = x;
      yy[i] = y;
      zz[i] = z;
    }
    for(const auto& cell: triangulation.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
          if(cell->face(f)->at_boundary()) {
            bool of_interest = true;
            for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
              unsigned int i = 0;
              while(i < 1035) {
                if(std::abs(cell->face(f)->vertex(v)(0) - xx[i]) < 1e-8 &&
                  std::abs(cell->face(f)->vertex(v)(1) - yy[i]) < 1e-8 &&
                  std::abs(cell->face(f)->vertex(v)(2) - zz[i]) < 1e-8)
                  break;
                 i++;
              }
              if(i == 1035) {
                of_interest = false;
                break;
              }
            }
            if(of_interest)
              cell->face(f)->set_boundary_id(2);
          }
        }
      }
    }

    pcout<<"Grid read"<<std::endl;
  }


  // After creating the triangulation, it creates the mesh dependent
  // data, i.e. it distributes degrees of freedom and renumbers them, and
  // initializes the matrices and vectors that we will use.
  //
  template<int dim>
  void NavierStokesProjection<dim>::setup_dofs() {
    Linfty_error_per_cell_vel.reinit(triangulation.n_active_cells());

    pcout<<"Boundary ids:"<<std::endl;
    const auto& boundary_ids = triangulation.get_boundary_ids();
    for(const auto& elem: boundary_ids)
      pcout<<elem<<std::endl;
    pcout << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl;
    pcout << "Number of levels: "       << triangulation.n_global_levels()       << std::endl;

    navier_stokes_matrix.clear();
    mg_matrices.clear_elements();

    dof_handler_velocity.distribute_dofs(fe_velocity);
    dof_handler_pressure.distribute_dofs(fe_pressure);

    dof_handler_velocity.distribute_mg_dofs();
    dof_handler_pressure.distribute_mg_dofs();

    pcout << "dim (X_h) = " << dof_handler_velocity.n_dofs()
          << std::endl
          << "dim (M_h) = " << dof_handler_pressure.n_dofs()
          << std::endl
          << "Re        = " << Re << std::endl
          << std::endl;

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
      output_n_dofs_velocity << dof_handler_velocity.n_dofs() << std::endl;
      output_n_dofs_pressure << dof_handler_pressure.n_dofs() << std::endl;
    }

    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                            update_quadrature_points | update_values);
    additional_data.mapping_update_flags_inner_faces = (update_gradients | update_JxW_values | update_quadrature_points |
                                                        update_normal_vectors | update_values);
    additional_data.mapping_update_flags_boundary_faces = (update_gradients | update_JxW_values | update_quadrature_points |
                                                           update_normal_vectors | update_values);
    additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;

    dof_handlers.clear();
    dof_handlers.push_back(&dof_handler_velocity);
    dof_handlers.push_back(&dof_handler_pressure);

    std::vector<const AffineConstraints<double>*> constraints;
    constraints.push_back(&constraints_velocity);
    constraints.push_back(&constraints_pressure);

    quadratures.clear();
    quadratures.push_back(QGauss<1>(EquationData::degree_p + 2));
    quadratures.push_back(QGauss<1>(EquationData::degree_p + 1));

    matrix_free_storage->reinit(dof_handlers, constraints, quadratures, additional_data);
    matrix_free_storage->initialize_dof_vector(u_star, 0);
    matrix_free_storage->initialize_dof_vector(rhs_u, 0);
    matrix_free_storage->initialize_dof_vector(u_n, 0);
    matrix_free_storage->initialize_dof_vector(u_extr, 0);
    matrix_free_storage->initialize_dof_vector(u_n_minus_1, 0);
    matrix_free_storage->initialize_dof_vector(u_n_gamma, 0);
    matrix_free_storage->initialize_dof_vector(u_tmp, 0);

    matrix_free_storage->initialize_dof_vector(u_n_k, 0);
    matrix_free_storage->initialize_dof_vector(u_n_gamma_k, 0);

    matrix_free_storage->initialize_dof_vector(pres_int, 1);
    matrix_free_storage->initialize_dof_vector(pres_n, 1);
    matrix_free_storage->initialize_dof_vector(rhs_p, 1);

    const unsigned int nlevels = triangulation.n_global_levels();
    mg_matrices.resize(0, nlevels - 1);
    for(unsigned int level = 0; level < nlevels; ++level) {
      mg_matrices[level].set_dt(dt);
      mg_matrices[level].set_Reynolds(Re);
    }
  }


  // @sect4{ <code>NavierStokesProjection::initialize</code> }

  // This method loads the initial data
  //
  template<int dim>
  void NavierStokesProjection<dim>::initialize() {
    TimerOutput::Scope t(time_table, "Initialize pressure and velocity");

    VectorTools::interpolate(dof_handler_pressure, pres_init, pres_n);

    VectorTools::interpolate(dof_handler_velocity, ZeroFunction<dim>(dim), u_n_minus_1);
    VectorTools::interpolate(dof_handler_velocity, ZeroFunction<dim>(dim), u_n);
  }


  // @sect4{<code>NavierStokesProjection::interpolate_velocity</code>}

  // This function computes the extrapolated velocity to be used in the momentum predictor
  //
  template<int dim>
  void NavierStokesProjection<dim>::interpolate_velocity() {
    TimerOutput::Scope t(time_table, "Interpolate velocity");

    //--- TR-BDF2 first step
    if(TR_BDF2_stage == 1) {
      u_extr.equ(1.0 + gamma/(2.0*(1.0 - gamma)), u_n);
      u_tmp.equ(gamma/(2.0*(1.0 - gamma)), u_n_minus_1);
      u_extr -= u_tmp;
    }
    //--- TR-BDF2 second step
    else {
      u_extr.equ(1.0 + (1.0 - gamma)/gamma, u_n_gamma);
      u_tmp.equ((1.0 - gamma)/gamma, u_n);
      u_extr -= u_tmp;
    }
  }


  // @sect4{<code>NavierStokesProjection::diffusion_step</code>}

  // The implementation of a diffusion step. Note that the expensive operation
  // is the diffusion solve at the end of the function, which we have to do once
  // for each velocity component. To accelerate things a bit, we allow to do
  // this in %parallel, using the Threads::new_task function which makes sure
  // that the <code>dim</code> solves are all taken care of and are scheduled to
  // available processors: if your machine has more than one processor core and
  // no other parts of this program are using resources currently, then the
  // diffusion solves will run in %parallel. On the other hand, if your system
  // has only one processor core then running things in %parallel would be
  // inefficient (since it leads, for example, to cache congestion) and things
  // will be executed sequentially.
  template<int dim>
  void NavierStokesProjection<dim>::diffusion_step() {
    TimerOutput::Scope t(time_table, "Diffusion step");

    const std::vector<unsigned int> tmp = {0};
    navier_stokes_matrix.initialize(matrix_free_storage, tmp, tmp);

    navier_stokes_matrix.set_NS_stage(1);

    if(TR_BDF2_stage == 1) {
      navier_stokes_matrix.vmult_rhs_velocity(rhs_u, {u_n, u_n, pres_n});
      u_n_k = u_n;
    }
    else {
      navier_stokes_matrix.vmult_rhs_velocity(rhs_u, {u_n, u_n_gamma, pres_int, u_n_gamma});
      u_n_gamma_k = u_n_gamma;
    }

    for(unsigned int iter = 0; iter < 1; ++iter) {
      if(TR_BDF2_stage == 1) {
        navier_stokes_matrix.set_u_extr(u_n_k);
        u_star = u_n_k;
      }
      else {
        navier_stokes_matrix.set_u_extr(u_n_gamma_k);
        u_star = u_n_gamma_k;
      }

      /*MGTransferMatrixFree<dim, float> mg_transfer;
      mg_transfer.build(dof_handler_velocity);

      MGLevelObject<LinearAlgebra::distributed::Vector<float>> level_projection(0, triangulation.n_global_levels() - 1);
      if(TR_BDF2_stage == 1)
        mg_transfer.interpolate_to_mg(dof_handler_velocity, level_projection, u_n_k);
      else
        mg_transfer.interpolate_to_mg(dof_handler_velocity, level_projection, u_n_gamma_k);

      for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
        typename MatrixFree<dim, float>::AdditionalData additional_data_mg;
        additional_data_mg.tasks_parallel_scheme = MatrixFree<dim, float>::AdditionalData::none;
        additional_data_mg.mapping_update_flags  = (update_gradients | update_JxW_values | update_values | update_quadrature_points);
        additional_data_mg.mapping_update_flags_inner_faces = (update_gradients | update_JxW_values | update_values |
                                                               update_quadrature_points | update_normal_vectors);
        additional_data_mg.mapping_update_flags_boundary_faces = (update_gradients | update_JxW_values | update_values | update_normal_vectors);
        additional_data_mg.mapping_update_flags  = (update_gradients | update_JxW_values | update_values);
        additional_data_mg.mapping_update_flags_inner_faces = (update_gradients | update_JxW_values | update_values);
        additional_data_mg.mapping_update_flags_boundary_faces = (update_gradients | update_JxW_values | update_values);
        additional_data_mg.mg_level = level;
        std::vector<const AffineConstraints<float>*> constraints_mg;
        AffineConstraints<float> constraints_velocity_mg;
        constraints_velocity_mg.clear();
        constraints_mg.push_back(&constraints_velocity_mg);
        AffineConstraints<float> constraints_pressure_mg;
        constraints_pressure_mg.clear();
        constraints_mg.push_back(&constraints_pressure_mg);
        std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level(new MatrixFree<dim, float>());
        mg_mf_storage_level->reinit(dof_handlers, constraints_mg, quadratures, additional_data_mg);
        mg_matrices[level].initialize(mg_mf_storage_level, tmp, tmp);
        mg_matrices[level].set_NS_stage(1);
        mg_matrices[level].set_u_extr(level_projection[level]);
        mg_matrices[level].compute_diagonal();
      }

      using SmootherType = PreconditionJacobi<NavierStokesProjectionOperator<dim,
                                                                             EquationData::degree_p,
                                                                             EquationData::degree_p + 1,
                                                                             EquationData::degree_p + 1,
                                                                             EquationData::degree_p + 2,
                                                                             LinearAlgebra::distributed::Vector<float>,
                                                                             float>>;
      MGSmootherPrecondition<NavierStokesProjectionOperator<dim,
                                                            EquationData::degree_p,
                                                            EquationData::degree_p + 1,
                                                            EquationData::degree_p + 1,
                                                            EquationData::degree_p + 2,
                                                            LinearAlgebra::distributed::Vector<float>,
                                                            float>,
                              SmootherType,
                              LinearAlgebra::distributed::Vector<float>> mg_smoother;
      mg_smoother.initialize(mg_matrices);

      PreconditionIdentity        identity;
      SolverGMRES<LinearAlgebra::distributed::Vector<float>>
      gmres_mg(solver_control, SolverGMRES<LinearAlgebra::distributed::Vector<float>>::AdditionalData(vel_Krylov_size));
      MGCoarseGridIterativeSolver<LinearAlgebra::distributed::Vector<float>,
                                  SolverGMRES<LinearAlgebra::distributed::Vector<float>>,
                                  NavierStokesProjectionOperator<dim,
                                                                 EquationData::degree_p,
                                                                 EquationData::degree_p + 1,
                                                                 EquationData::degree_p + 1,
                                                                 EquationData::degree_p + 2,
                                                                 LinearAlgebra::distributed::Vector<float>,
                                                                 float>,
                                  PreconditionIdentity> mg_coarse(gmres_mg, mg_matrices[0], identity);

      mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(mg_matrices);

      Multigrid<LinearAlgebra::distributed::Vector<float>> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);

      PreconditionMG<dim,
                     LinearAlgebra::distributed::Vector<float>,
                     MGTransferMatrixFree<dim, float>> preconditioner(dof_handler_velocity, mg, mg_transfer);*/

      PreconditionJacobi<NavierStokesProjectionOperator<dim,
                                                        EquationData::degree_p,
                                                        EquationData::degree_p + 1,
                                                        EquationData::degree_p + 1,
                                                        EquationData::degree_p + 2,
                                                        LinearAlgebra::distributed::Vector<double>,
                                                        double>> preconditioner;
      navier_stokes_matrix.compute_diagonal();
      preconditioner.initialize(navier_stokes_matrix);

      SolverControl solver_control(vel_max_its, vel_eps*rhs_u.l2_norm());
      SolverGMRES<LinearAlgebra::distributed::Vector<double>>
      gmres(solver_control, SolverGMRES<LinearAlgebra::distributed::Vector<double>>::AdditionalData(vel_Krylov_size));
      gmres.solve(navier_stokes_matrix, u_star, rhs_u, preconditioner);

      //Compute the relative error
      VectorTools::integrate_difference(dof_handler_velocity, u_star, ZeroFunction<dim>(dim),
                                        Linfty_error_per_cell_vel, quadrature_velocity, VectorTools::Linfty_norm);
      const double den = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_vel, VectorTools::Linfty_norm);
      double error = 0.0;
      u_tmp = u_star;
      if(TR_BDF2_stage == 1) {
        u_tmp -= u_n_k;
        VectorTools::integrate_difference(dof_handler_velocity, u_tmp, ZeroFunction<dim>(dim),
                                          Linfty_error_per_cell_vel, quadrature_velocity, VectorTools::Linfty_norm);
        error = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_vel, VectorTools::Linfty_norm)/den;
        u_n_k = u_star;
      }
      else {
        u_tmp -= u_n_gamma_k;
        VectorTools::integrate_difference(dof_handler_velocity, u_tmp, ZeroFunction<dim>(dim),
                                          Linfty_error_per_cell_vel, quadrature_velocity, VectorTools::Linfty_norm);
        error = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_vel, VectorTools::Linfty_norm)/den;
        u_n_gamma_k = u_star;
      }
      if(error < 1e-6)
        break;
    }
  }


  // @sect4{<code>NavierStokesProjection::projection_step</code>}

  // This implements the projection step
  //
  template<int dim>
  void NavierStokesProjection<dim>::projection_step() {
    TimerOutput::Scope t(time_table, "Projection step pressure");

    const std::vector<unsigned int> tmp = {1};
    navier_stokes_matrix.initialize(matrix_free_storage, tmp, tmp);
    if(TR_BDF2_stage == 1)
      navier_stokes_matrix.vmult_rhs_pressure(rhs_p, {u_star, pres_n});
    else
      navier_stokes_matrix.vmult_rhs_pressure(rhs_p, {u_star, pres_int});

    navier_stokes_matrix.set_NS_stage(2);

    SolverControl solver_control(vel_max_its, vel_eps*rhs_p.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

    for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
      typename MatrixFree<dim, float>::AdditionalData additional_data_mg;
      additional_data_mg.tasks_parallel_scheme = MatrixFree<dim, float>::AdditionalData::none;
      additional_data_mg.mapping_update_flags  = (update_gradients | update_JxW_values | update_values);
      additional_data_mg.mapping_update_flags_inner_faces = (update_gradients | update_JxW_values | update_values);
      additional_data_mg.mapping_update_flags_boundary_faces = (update_gradients | update_JxW_values);
      additional_data_mg.mg_level = level;
      std::vector<const AffineConstraints<float>*> constraints_mg;
      AffineConstraints<float> constraints_velocity_mg;
      constraints_velocity_mg.clear();
      constraints_mg.push_back(&constraints_velocity_mg);
      AffineConstraints<float> constraints_pressure_mg;
      constraints_pressure_mg.clear();
      constraints_mg.push_back(&constraints_pressure_mg);
      std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level(new MatrixFree<dim, float>());
      mg_mf_storage_level->reinit(dof_handlers, constraints_mg, quadratures, additional_data_mg);
      mg_matrices[level].initialize(mg_mf_storage_level, tmp, tmp);
      mg_matrices[level].set_NS_stage(2);
    }

    MGTransferMatrixFree<dim, float> mg_transfer;
    mg_transfer.build(dof_handler_pressure);

    using SmootherType = PreconditionChebyshev<NavierStokesProjectionOperator<dim,
                                                                              EquationData::degree_p,
                                                                              EquationData::degree_p + 1,
                                                                              EquationData::degree_p + 1,
                                                                              EquationData::degree_p + 2,
                                                                              LinearAlgebra::distributed::Vector<float>,
                                                                              float>,
                                               LinearAlgebra::distributed::Vector<float>>;
    mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<float>> mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);
    for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
      if(level > 0) {
        smoother_data[level].smoothing_range     = 15.0;
        smoother_data[level].degree              = 3;
        smoother_data[level].eig_cg_n_iterations = 10;
      }
      else {
        smoother_data[0].smoothing_range = 2e-2;
        smoother_data[0].degree          = numbers::invalid_unsigned_int;
        smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
      }
      mg_matrices[level].compute_diagonal();
      smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
    }
    mg_smoother.initialize(mg_matrices, smoother_data);

    PreconditionIdentity                                identity;
    SolverCG<LinearAlgebra::distributed::Vector<float>> cg_mg(solver_control);
    MGCoarseGridIterativeSolver<LinearAlgebra::distributed::Vector<float>,
                                SolverCG<LinearAlgebra::distributed::Vector<float>>,
                                NavierStokesProjectionOperator<dim,
                                                               EquationData::degree_p,
                                                               EquationData::degree_p + 1,
                                                               EquationData::degree_p + 1,
                                                               EquationData::degree_p + 2,
                                                               LinearAlgebra::distributed::Vector<float>,
                                                               float>,
                                PreconditionIdentity> mg_coarse(cg_mg, mg_matrices[0], identity);

    mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(mg_matrices);

    Multigrid<LinearAlgebra::distributed::Vector<float>> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);

    PreconditionMG<dim,
                   LinearAlgebra::distributed::Vector<float>,
                   MGTransferMatrixFree<dim, float>> preconditioner(dof_handler_pressure, mg, mg_transfer);


    /*PreconditionJacobi<NavierStokesProjectionOperator<dim,
                                                      EquationData::degree_p,
                                                      EquationData::degree_p + 1,
                                                      EquationData::degree_p + 1,
                                                      EquationData::degree_p + 2,
                                                      LinearAlgebra::distributed::Vector<double>,
                                                      double>> preconditioner;
    navier_stokes_matrix.compute_diagonal();
    preconditioner.initialize(navier_stokes_matrix);*/

    if(TR_BDF2_stage == 1) {
      pres_int = pres_n;
      cg.solve(navier_stokes_matrix, pres_int, rhs_p, preconditioner);
    }
    else {
      pres_n = pres_int;
      cg.solve(navier_stokes_matrix, pres_n, rhs_p, preconditioner);
    }
  }


  // This implements the projection step for the gradient of pressure
  //
  template<int dim>
  void NavierStokesProjection<dim>::project_grad(const unsigned int flag) {
    TimerOutput::Scope t(time_table, "Gradient of pressure projection");

    AssertIndexRange(flag, 3);
    Assert(flag > 0, ExcInternalError());
    const std::vector<unsigned int> tmp = {0};
    navier_stokes_matrix.initialize(matrix_free_storage, tmp, tmp);
    if(flag == 1)
      navier_stokes_matrix.vmult_grad_p_projection(rhs_u, pres_n);
    else if(flag == 2)
      navier_stokes_matrix.vmult_grad_p_projection(rhs_u, pres_int);

    navier_stokes_matrix.set_NS_stage(3);

    SolverControl solver_control(vel_max_its, 1e-12*rhs_u.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
    cg.solve(navier_stokes_matrix, u_tmp, rhs_u, PreconditionIdentity());
  }


  // The following function is used in determining the maximal velocity
  // in order to compute the CFL
  //
  template<int dim>
  double NavierStokesProjection<dim>::get_maximal_velocity() {
    VectorTools::integrate_difference(dof_handler_velocity, u_n, ZeroFunction<dim>(dim),
                                      Linfty_error_per_cell_vel, quadrature_velocity, VectorTools::Linfty_norm);
    const double res = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_vel, VectorTools::Linfty_norm);

    return res;
  }


  // The following function is used in determining the maximal nodal difference
  // in order to see if we have reched steady-state
  template<int dim>
  double NavierStokesProjection<dim>::get_maximal_difference() {
    u_tmp = u_n;
    u_tmp -= u_n_minus_1;
    VectorTools::integrate_difference(dof_handler_velocity, u_tmp, ZeroFunction<dim>(dim),
                                      Linfty_error_per_cell_vel, quadrature_velocity, VectorTools::Linfty_norm);
    const double res = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_vel, VectorTools::Linfty_norm);
    pcout << "Maximum nodal difference = " << res <<std::endl;

    return res;
  }


  // @sect4{ <code>NavierStokesProjection::output_results</code> }

  // This method plots the current solution. The main difficulty is that we want
  // to create a single output file that contains the data for all velocity
  // components and the pressure. On the other hand, velocities and the pressure
  // live on separate DoFHandler objects, and
  // so can't be written to the same file using a single DataOut object. As a
  // consequence, we have to work a bit harder to get the various pieces of data
  // into a single DoFHandler object, and then use that to drive graphical
  // output.
  //
  template<int dim>
  void NavierStokesProjection<dim>::output_results(const unsigned int step) {
    TimerOutput::Scope t(time_table, "Output results");

    DataOut<dim> data_out;
    std::vector<std::string> velocity_names(dim, "v");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation_velocity(dim, DataComponentInterpretation::component_is_part_of_vector);
    u_n.update_ghost_values();
    data_out.add_data_vector(dof_handler_velocity, u_n, velocity_names, component_interpretation_velocity);
    pres_n.update_ghost_values();
    data_out.add_data_vector(dof_handler_pressure, pres_n, "p", {DataComponentInterpretation::component_is_scalar});

    std::vector<std::string> velocity_names_old(dim, "v_old");
    u_n_minus_1.update_ghost_values();
    data_out.add_data_vector(dof_handler_velocity, u_n_minus_1, velocity_names_old, component_interpretation_velocity);

    PostprocessorVorticity<dim> postprocessor;
    data_out.add_data_vector(dof_handler_velocity, u_n, postprocessor);

    data_out.build_patches();
    const std::string output = "./" + saving_dir + "/solution-" + Utilities::int_to_string(step, 5) + ".vtu";
    data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);
  }


  // @sect4{ <code>NavierStokesProjection::refine_mesh</code>}
  //
  // After finding a good initial guess on the coarse mesh, we hope to
  // decrease the error through refining the mesh. We also need to transfer the current solution to the
  // next mesh using the SolutionTransfer class.
  //
  template <int dim>
  void NavierStokesProjection<dim>::refine_mesh() {
    TimerOutput::Scope t(time_table, "Refine mesh");

    //Save current velocity solution
    LinearAlgebra::distributed::Vector<double> tmp_velocity;
    tmp_velocity.reinit(u_n);
    tmp_velocity = u_n;

    //Build a proper vector for KellyErrorEstimator
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler_velocity, locally_relevant_dofs);
    LinearAlgebra::distributed::Vector<double> copy_vec(tmp_velocity);
    tmp_velocity.reinit(dof_handler_velocity.locally_owned_dofs(), locally_relevant_dofs, MPI_COMM_WORLD);
    tmp_velocity.copy_locally_owned_data_from(copy_vec);
    tmp_velocity.update_ghost_values();

    //Call KellyErrorEstimator
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(dof_handler_velocity,
                                       QGauss<dim - 1>(EquationData::degree_p + 1),
                                       std::map<types::boundary_id, const Function<dim>*>(),
                                       tmp_velocity,
                                       estimated_error_per_cell);

    //Refine grid
    GridRefinement::refine_and_coarsen_fixed_number(triangulation, estimated_error_per_cell, 0.3, 0.05);
    for(const auto& cell: triangulation.active_cell_iterators()) {
      if(cell->refine_flag_set() && cell->level() == max_loc_refinements)
        cell->clear_refine_flag();
      if(cell->coarsen_flag_set() && cell->level() == min_loc_refinements)
        cell->clear_coarsen_flag();
    }
    triangulation.prepare_coarsening_and_refinement();

    std::vector<const LinearAlgebra::distributed::Vector<double>*> velocities;
    velocities.push_back(&u_n);
    velocities.push_back(&u_n_minus_1);
    parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>
    solution_transfer_velocity(dof_handler_velocity);
    solution_transfer_velocity.prepare_for_coarsening_and_refinement(velocities);
    parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>
    solution_transfer_pressure(dof_handler_pressure);
    solution_transfer_pressure.prepare_for_coarsening_and_refinement(pres_n);

    triangulation.execute_coarsening_and_refinement();

    // First DoFHandler objects are set up and constraints are generated.
    setup_dofs();

    // Interpolate current solutions to new mesh
    LinearAlgebra::distributed::Vector<double> transfer_velocity,
                                               transfer_velocity_minus_1,
                                               transfer_pressure;
    transfer_velocity.reinit(u_n);
    transfer_velocity.zero_out_ghosts();
    transfer_velocity_minus_1.reinit(u_n_minus_1);
    transfer_velocity_minus_1.zero_out_ghosts();
    transfer_pressure.reinit(pres_n);
    transfer_pressure.zero_out_ghosts();

    std::vector<LinearAlgebra::distributed::Vector<double>*> transfer_velocities;
    transfer_velocities.push_back(&transfer_velocity);
    transfer_velocities.push_back(&transfer_velocity_minus_1);
    solution_transfer_velocity.interpolate(transfer_velocities);
    transfer_velocity.update_ghost_values();
    transfer_velocity_minus_1.update_ghost_values();
    solution_transfer_pressure.interpolate(transfer_pressure);
    transfer_pressure.update_ghost_values();

    u_n         = transfer_velocity;
    u_n_minus_1 = transfer_velocity_minus_1;
    pres_n      = transfer_pressure;
  }


  // @sect4{ <code>NavierStokesProjection::interpolate_max_res</code>}
  //
  // Interpolate the locally refined solution to a mesh with maximal resolution
  // and transfer velocity and pressure.
  //
  template<int dim>
  void NavierStokesProjection<dim>::interpolate_max_res(const unsigned int level) {
    parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>
    solution_transfer_velocity(dof_handler_velocity);
    std::vector<const LinearAlgebra::distributed::Vector<double>*> velocities;
    velocities.push_back(&u_n);
    velocities.push_back(&u_n_minus_1);
    solution_transfer_velocity.prepare_for_coarsening_and_refinement(velocities);

    parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>
    solution_transfer_pressure(dof_handler_pressure);
    solution_transfer_pressure.prepare_for_coarsening_and_refinement(pres_n);

    for(const auto& cell: triangulation.active_cell_iterators_on_level(level)) {
      if(cell->is_locally_owned())
        cell->set_refine_flag();
    }

    triangulation.execute_coarsening_and_refinement();

    setup_dofs();

    LinearAlgebra::distributed::Vector<double> transfer_velocity, transfer_velocity_minus_1,
                                               transfer_pressure;

    transfer_velocity.reinit(u_n);
    transfer_velocity.zero_out_ghosts();
    transfer_velocity_minus_1.reinit(u_n_minus_1);
    transfer_velocity_minus_1.zero_out_ghosts();

    transfer_pressure.reinit(pres_n);
    transfer_pressure.zero_out_ghosts();

    std::vector<LinearAlgebra::distributed::Vector<double>*> transfer_velocities;

    transfer_velocities.push_back(&transfer_velocity);
    transfer_velocities.push_back(&transfer_velocity_minus_1);
    solution_transfer_velocity.interpolate(transfer_velocities);
    transfer_velocity.update_ghost_values();
    transfer_velocity_minus_1.update_ghost_values();

    solution_transfer_pressure.interpolate(transfer_pressure);
    transfer_pressure.update_ghost_values();

    u_n            = transfer_velocity;
    u_n_minus_1    = transfer_velocity_minus_1;
    pres_n         = transfer_pressure;
  }


  // @sect4{ <code>NavierStokesProjection::save_max_res</code>}
  //
  // Save maximum resolution to a mesh adapted for paraview
  // in order to perform the difference
  //
  template<int dim>
  void NavierStokesProjection<dim>::save_max_res() {
    parallel::distributed::Triangulation<dim> triangulation_tmp(MPI_COMM_WORLD);
    const double Gamma = 1.0;
    const double Lambda = 1.0;
    GridGenerator::subdivided_hyper_rectangle(triangulation_tmp, {6, 6, 6},
                                              Point<dim>(-0.5*Gamma, -0.5, -0.5*Lambda),
                                              Point<dim>(0.5*Gamma, 0.5, 0.5*Lambda), true);
    triangulation_tmp.refine_global(triangulation.n_global_levels() - 1);
    DoFHandler<dim> dof_handler_velocity_tmp(triangulation_tmp);
    DoFHandler<dim> dof_handler_pressure_tmp(triangulation_tmp);
    dof_handler_velocity_tmp.distribute_dofs(fe_velocity);
    dof_handler_pressure_tmp.distribute_dofs(fe_pressure);

    LinearAlgebra::distributed::Vector<double> u_n_tmp, u_n_minus_1_tmp,
                                               pres_n_tmp;
    u_n_tmp.reinit(dof_handler_velocity_tmp.n_dofs());
    u_n_minus_1_tmp.reinit(dof_handler_velocity_tmp.n_dofs());
    pres_n_tmp.reinit(dof_handler_pressure_tmp.n_dofs());

    DataOut<dim> data_out;
    std::vector<std::string> velocity_names(dim, "v");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation_velocity(dim, DataComponentInterpretation::component_is_part_of_vector);
    VectorTools::interpolate_to_different_mesh(dof_handler_velocity, u_n, dof_handler_velocity_tmp, u_n_tmp);
    u_n_tmp.update_ghost_values();
    data_out.add_data_vector(dof_handler_velocity_tmp, u_n_tmp, velocity_names, component_interpretation_velocity);
    VectorTools::interpolate_to_different_mesh(dof_handler_pressure, pres_n, dof_handler_pressure_tmp, pres_n_tmp);
    pres_n_tmp.update_ghost_values();
    data_out.add_data_vector(dof_handler_pressure_tmp, pres_n_tmp, "p", {DataComponentInterpretation::component_is_scalar});
    PostprocessorVorticity<dim> postprocessor;
    data_out.add_data_vector(dof_handler_velocity_tmp, u_n_tmp, postprocessor);
    data_out.build_patches();
    const std::string output = "./" + saving_dir + "/solution_max_res_end.vtu";
    data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);

    DataOut<dim> data_out_old;
    VectorTools::interpolate_to_different_mesh(dof_handler_velocity, u_n_minus_1, dof_handler_velocity_tmp, u_n_minus_1_tmp);
    u_n_minus_1_tmp.update_ghost_values();
    data_out_old.add_data_vector(dof_handler_velocity_tmp, u_n_minus_1_tmp, velocity_names, component_interpretation_velocity);
    PostprocessorVorticity<dim> postprocessor_old;
    data_out_old.add_data_vector(dof_handler_velocity_tmp, u_n_minus_1_tmp, postprocessor_old);
    data_out_old.build_patches();
    const std::string output_old = "./" + saving_dir + "/solution_max_res_end_minus_1.vtu";
    data_out_old.write_vtu_in_parallel(output_old, MPI_COMM_WORLD);
  }


  // @sect4{ <code>NavierStokesProjection::run</code> }

  // This is the time marching function, which starting at <code>t_0</code>
  // advances in time using the projection method with time step <code>dt</code>
  // until <code>T</code>.
  //
  // Its second parameter, <code>verbose</code> indicates whether the function
  // should output information what it is doing at any given moment:
  // we use the ConditionalOStream class to do that for us.
  //
  template<int dim>
  void NavierStokesProjection<dim>::run(const bool verbose, const unsigned int output_interval) {
    ConditionalOStream verbose_cout(std::cout, verbose && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    output_results(1);
    double time = t_0 + dt;
    unsigned int n = 1;
    while(std::abs(T - time) > 1e-10) {
      time += dt;
      n++;
      pcout << "Step = " << n << " Time = " << time << std::endl;
      //--- First stage of TR-BDF2
      navier_stokes_matrix.set_TR_BDF2_stage(TR_BDF2_stage);
      for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
        mg_matrices[level].set_TR_BDF2_stage(TR_BDF2_stage);
      /*verbose_cout << "  Interpolating the velocity stage 1" << std::endl;
      interpolate_velocity();*/
      verbose_cout << "  Diffusion Step stage 1 " << std::endl;
      diffusion_step();
      verbose_cout << "  Projection Step stage 1" << std::endl;
      if(n > 2)
        project_grad(1);
      else
        u_tmp = 0.0;
      u_tmp.equ(gamma*dt, u_tmp);
      u_star += u_tmp;
      projection_step();
      verbose_cout << "  Updating the Velocity stage 1" << std::endl;
      u_n_gamma.equ(1.0, u_star);
      project_grad(2);
      u_tmp.equ(-gamma*dt, u_tmp);
      u_n_gamma += u_tmp;
      u_n_minus_1 = u_n;
      TR_BDF2_stage = 2; //--- Flag to pass at second stage
      //--- Second stage of TR-BDF2
      navier_stokes_matrix.set_TR_BDF2_stage(TR_BDF2_stage);
      for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
        mg_matrices[level].set_TR_BDF2_stage(TR_BDF2_stage);
      /*verbose_cout << "  Interpolating the velocity stage 2" << std::endl;
      interpolate_velocity();*/
      verbose_cout << "  Diffusion Step stage 2 " << std::endl;
      diffusion_step();
      verbose_cout << "  Projection Step stage 2" << std::endl;
      project_grad(2);
      u_tmp.equ((1.0 - gamma)*dt, u_tmp);
      u_star += u_tmp;
      projection_step();
      verbose_cout << "  Updating the Velocity stage 2" << std::endl;
      u_n.equ(1.0, u_star);
      project_grad(1);
      u_tmp.equ((gamma - 1.0)*dt, u_tmp);
      u_n += u_tmp;
      TR_BDF2_stage = 1; //--- Flag to pass at first stage at next step
      const double max_vel = get_maximal_velocity();
      pcout<< "Maximal velocity = " << max_vel << std::endl;
      pcout << "CFL = " << dt*max_vel*(EquationData::degree_p + 1)*
                           std::sqrt(dim)/GridTools::minimal_cell_diameter(triangulation) << std::endl;
      if(n % output_interval == 0) {
        verbose_cout << "Plotting Solution final" << std::endl;
        output_results(n);
      }
      if(time > 0.001*T && get_maximal_difference()/dt < 1e-7)
        break;
      else if(T - time < dt && T - time > 1e-10) {
        dt = T - time;
        navier_stokes_matrix.set_dt(dt);
        for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
          mg_matrices[level].set_dt(dt);
      }
      if(refinement_iterations > 0 && n % refinement_iterations == 0) {
        verbose_cout << "Refining mesh" << std::endl;
        refine_mesh();
      }
    }
    if(n % output_interval != 0) {
      verbose_cout << "Plotting Solution final" << std::endl;
      output_results(n);
    }
    if(refinement_iterations > 0) {
      for(unsigned int lev = 0; lev < triangulation.n_global_levels() - 1; ++ lev)
        interpolate_max_res(lev);
      save_max_res();
    }
  }

} // namespace Step35


// @sect3{ The main function }

// The main function looks very much like in all the other tutorial programs, so
// there is little to comment on here:
int main(int argc, char *argv[]) {
  try {
    using namespace Step35;

    RunTimeParameters::Data_Storage data;
    data.read_data("parameter-file.prm");

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, -1);

    const auto& curr_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    deallog.depth_console(data.verbose && curr_rank == 0 ? 2 : 0);

    NavierStokesProjection<3> test(data);
    test.run(data.verbose, data.output_interval);

    if(curr_rank == 0)
      std::cout << "----------------------------------------------------"
                << std::endl
                << "Apparently everything went fine!" << std::endl
                << "Don't forget to brush your teeth :-)" << std::endl
                << std::endl;

    return 0;
  }
  catch(std::exception &exc) {
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
  catch(...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

}

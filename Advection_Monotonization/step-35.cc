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

      double dt;

      unsigned int n_global_refines;
      unsigned int n_cells;
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
                                  dt(5e-4),
                                  n_global_refines(0),
                                  n_cells(0),
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
                          "0.0",
                          Patterns::Double(0.0),
                          " The initial time of the simulation. ");
        prm.declare_entry("final_time",
                          "1.0",
                          Patterns::Double(0.0),
                          " The final time of the simulation. ");
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
        prm.declare_entry("n_of_refines",
                          "3",
                          Patterns::Integer(1, 15),
                          " The number of global refinements we want for the mesh. ");
        prm.declare_entry("n_of_cells",
                          "100",
                          Patterns::Integer(1, 1500),
                          " The number of cells we want on each direction of the mesh. ");
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
      }
      prm.leave_subsection();

      prm.enter_subsection("Time step data");
      {
        dt = prm.get_double("dt");
      }
      prm.leave_subsection();

      prm.enter_subsection("Space discretization");
      {
        n_global_refines = prm.get_integer("n_of_refines");
        n_cells = prm.get_integer("n_of_cells");
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


  template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
  }


  // @sect3{Equation data}

  // In the next namespace, we declare the initial and boundary conditions
  //
  namespace EquationData {
    static const unsigned int degree = 1;

    // With this class defined, we declare class that describes the initial
    // condition for velocity:
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
      AssertIndexRange(component, 1);

      return 1.0;
    }


    template<int dim>
    void Velocity<dim>::vector_value(const Point<dim>& p, Vector<double>& values) const {
      Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));
      for(unsigned int i = 0; i < dim; ++i)
        values[i] = value(p, i);
    }

    //We do the same for the density
    //
    template<int dim>
    class Density: public Function<dim> {
    public:
      Density(const double initial_time = 0.0);

      virtual double value(const Point<dim>&  p,
                           const unsigned int component = 0) const override;
    };


    template<int dim>
    Density<dim>::Density(const double initial_time): Function<dim>(1, initial_time) {}


    template<int dim>
    double Density<dim>::value(const Point<dim>& p, const unsigned int component) const {
      (void)component;
      AssertIndexRange(component, 1);

      return 0.5*(sgn(std::sin(numbers::PI*(p[0] + 0.5 - this->get_time()))) + 1.0)*
             std::pow(std::cos(numbers::PI*(p[0] - this->get_time())), 4);
    }

  } // namespace EquationData


  // @sect3{ <code>HYPERBOLICOperator::HYPERBOLICOperator</code> }
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  class HYPERBOLICOperator: public MatrixFreeOperators::Base<dim, Vec> {
  public:
    HYPERBOLICOperator();

    HYPERBOLICOperator(RunTimeParameters::Data_Storage& data);

    void set_dt(const double time_step);

    void set_HYPERBOLIC_stage(const unsigned int stage);

    void set_NS_stage(const unsigned int stage);

    void advance_boundary_time(const double advance_dt);

    void vmult_rhs(Vec& dst, const std::vector<Vec>& src) const;

    void vmult_rhs_Q0(Vec& dst, const std::vector<Vec>& src) const;

    virtual void compute_diagonal() override {}

  protected:
    EquationData::Density<dim> rho_boundary;
    EquationData::Velocity<dim> u;

    double       dt;

    double       gamma;
    double       a21;
    double       a22;
    double       a31;
    double       a32;
    double       a33;

    unsigned int HYPERBOLIC_stage;
    unsigned int NS_stage;

    virtual void apply_add(Vec& dst, const Vec& src) const override;

  private:
    mutable Tensor<1, dim, VectorizedArray<Number>> velocity;

    void assemble_rhs_cell_term_rho_projection(const MatrixFree<dim, Number>&               data,
                                               Vec&                                         dst,
                                               const std::vector<Vec>&                      src,
                                               const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_rho_projection(const MatrixFree<dim, Number>&               data,
                                               Vec&                                         dst,
                                               const std::vector<Vec>&                      src,
                                               const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_rho_projection(const MatrixFree<dim, Number>&               data,
                                                   Vec&                                         dst,
                                                   const std::vector<Vec>&                      src,
                                                   const std::pair<unsigned int, unsigned int>& face_range) const {}

    void assemble_cell_term_rho_projection(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const Vec&                                   src,
                                           const std::pair<unsigned int, unsigned int>& cell_range) const;

    void assemble_rhs_cell_term_rho_projection_Q0(const MatrixFree<dim, Number>&               data,
                                                  Vec&                                         dst,
                                                  const std::vector<Vec>&                      src,
                                                  const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_rho_projection_Q0(const MatrixFree<dim, Number>&               data,
                                                  Vec&                                         dst,
                                                  const std::vector<Vec>&                      src,
                                                  const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_rho_projection_Q0(const MatrixFree<dim, Number>&               data,
                                                      Vec&                                         dst,
                                                      const std::vector<Vec>&                      src,
                                                      const std::pair<unsigned int, unsigned int>& face_range) const {}

    void assemble_cell_term_rho_projection_Q0(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const Vec&                                   src,
                                              const std::pair<unsigned int, unsigned int>& cell_range) const;
  };


  // Default constructor
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  HYPERBOLICOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  HYPERBOLICOperator(): MatrixFreeOperators::Base<dim, Vec>(), rho_boundary(), u(), dt(), gamma(2.0 - std::sqrt(2.0)),
                        a21(gamma), a22(0.0), a31((2.0*gamma - 1.0)/6.0), a32((7.0 - 2.0*gamma)/6.0), a33(0.0),
                        HYPERBOLIC_stage(1), NS_stage(1), velocity() {}


  // Constructor with runtime parameters storage
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  HYPERBOLICOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  HYPERBOLICOperator(RunTimeParameters::Data_Storage& data): MatrixFreeOperators::Base<dim, Vec>(), rho_boundary(data.initial_time),
                                                             u(data.initial_time), dt(data.dt), gamma(2.0 - std::sqrt(2.0)),
                                                             a21(gamma), a22(0.0), a31((2.0*gamma - 1.0)/6.0),
                                                             a32((7.0 - 2.0*gamma)/6.0), a33(0.0),
                                                             HYPERBOLIC_stage(1), NS_stage(1), velocity() {}


  // Setter of time-step
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  set_dt(const double time_step) {
    dt = time_step;
  }


  // Setter of HYPERBOLIC stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  set_HYPERBOLIC_stage(const unsigned int stage) {
    AssertIndexRange(stage, 3);
    Assert(stage > 0, ExcInternalError());
    HYPERBOLIC_stage = stage;
  }


  // Setter of HYPERBOLIC stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  set_NS_stage(const unsigned int stage) {
    AssertIndexRange(stage, 3);
    Assert(stage > 0, ExcInternalError());
    NS_stage = stage;
  }


  // Advance boundary time
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  advance_boundary_time(const double advance_dt) {
    rho_boundary.advance_time(advance_dt);
  }


  // Assemble rhs cell term for the density update
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  assemble_rhs_cell_term_rho_projection(const MatrixFree<dim, Number>&               data,
                                        Vec&                                         dst,
                                        const std::vector<Vec>&                      src,
                                        const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number>   phi(data, 1),
                                                               phi_rho_old(data, 1);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi.reinit(cell);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& point_vectorized = phi.quadrature_point(q);
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d)
              point[d] = point_vectorized[d][v];
            for(unsigned int d = 0; d < dim; ++d)
              velocity[d][v] = u.value(point, d);
          }

          phi.submit_value(phi_rho_old.get_value(q), q);
          phi.submit_gradient(a21*dt*phi_rho_old.get_value(q)*velocity, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number>   phi(data, 1),
                                                               phi_rho_tmp_2(data, 1);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[0], true, false);
        phi.reinit(cell);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& point_vectorized = phi.quadrature_point(q);
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d)
              point[d] = point_vectorized[d][v];
            for(unsigned int d = 0; d < dim; ++d)
              velocity[d][v] = u.value(point, d);
          }

          phi.submit_value(phi_rho_tmp_2.get_value(q), q);
          phi.submit_gradient(a32*a21/a31*dt*phi_rho_tmp_2.get_value(q)*velocity, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
  }


  // Assemble rhs face term for the density update
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  assemble_rhs_face_term_rho_projection(const MatrixFree<dim, Number>&               data,
                                        Vec&                                         dst,
                                        const std::vector<Vec>&                      src,
                                        const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, Number>   phi_p(data, true, 1),
                                                                   phi_m(data, false, 1),
                                                                   phi_rho_old_p(data, true, 1),
                                                                   phi_rho_old_m(data, false, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_p.reinit(face);
        phi_m.reinit(face);

        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& point_vectorized = phi_p.quadrature_point(q);
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d)
              point[d] = point_vectorized[d][v];
            for(unsigned int d = 0; d < dim; ++d)
              velocity[d][v] = u.value(point, d);
          }

          const auto& n_plus       = phi_p.get_normal_vector(q);

          const auto& avg_flux     = 0.5*(phi_rho_old_p.get_value(q)*velocity +
                                          phi_rho_old_m.get_value(q)*velocity);
          const auto  lambda_old   = std::max(std::abs(scalar_product(velocity, n_plus)),
                                              std::abs(scalar_product(velocity, n_plus)));
          const auto& jump_rho_old = phi_rho_old_p.get_value(q) - phi_rho_old_m.get_value(q);

          phi_p.submit_value(-a21*dt*(scalar_product(avg_flux, n_plus) + 0.5*lambda_old*jump_rho_old), q);
          phi_m.submit_value(a21*dt*(scalar_product(avg_flux, n_plus) + 0.5*lambda_old*jump_rho_old), q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, Number>   phi_p(data, true, 1),
                                                                   phi_m(data, false, 1),
                                                                   phi_rho_tmp_2_p(data, true, 1),
                                                                   phi_rho_tmp_2_m(data, false, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_tmp_2_p.reinit(face);
        phi_rho_tmp_2_p.gather_evaluate(src[0], true, false);
        phi_rho_tmp_2_m.reinit(face);
        phi_rho_tmp_2_m.gather_evaluate(src[0], true, false);
        phi_p.reinit(face);
        phi_m.reinit(face);

        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& point_vectorized = phi_p.quadrature_point(q);
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d)
              point[d] = point_vectorized[d][v];
            for(unsigned int d = 0; d < dim; ++d)
              velocity[d][v] = u.value(point, d);
          }

          const auto& n_plus         = phi_p.get_normal_vector(q);

          const auto& avg_flux_tmp_2 = 0.5*(phi_rho_tmp_2_p.get_value(q)*velocity +
                                            phi_rho_tmp_2_m.get_value(q)*velocity);
          const auto  lambda_tmp_2   = std::max(std::abs(scalar_product(velocity, n_plus)),
                                                std::abs(scalar_product(velocity, n_plus)));
          const auto& jump_rho_tmp_2 = phi_rho_tmp_2_p.get_value(q) - phi_rho_tmp_2_m.get_value(q);

          phi_p.submit_value(-a32*a21/a31*dt*(scalar_product(avg_flux_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_tmp_2), q);
          phi_m.submit_value(a32*a21/a31*dt*(scalar_product(avg_flux_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_tmp_2), q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
  }


  // Put together all the previous steps for density update
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  vmult_rhs(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d)
      src[d].update_ghost_values();

    this->data->loop(&HYPERBOLICOperator::assemble_rhs_cell_term_rho_projection,
                     &HYPERBOLICOperator::assemble_rhs_face_term_rho_projection,
                     &HYPERBOLICOperator::assemble_rhs_boundary_term_rho_projection,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble cell term for the density update
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  assemble_cell_term_rho_projection(const MatrixFree<dim, Number>&               data,
                                    Vec&                                         dst,
                                    const Vec&                                   src,
                                    const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi(data, 1);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, true, false);

      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(phi.get_value(q), q);

      phi.integrate_scatter(true, false, dst);
    }
  }


  // Assemble rhs Q0 cell term for the density update
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  assemble_rhs_cell_term_rho_projection_Q0(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const std::vector<Vec>&                      src,
                                           const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEEvaluation<dim, 0, n_q_points_1d, 1, Number>   phi(data, 2),
                                                       phi_rho_old(data, 2);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi.reinit(cell);

        for(unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_value(phi_rho_old.get_value(q), q);

        phi.integrate_scatter(true, false, dst);
      }
    }
    else {
      FEEvaluation<dim, 0, n_q_points_1d, 1, Number>   phi(data, 2),
                                                       phi_rho_tmp_2(data, 2);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[0], true, false);
        phi.reinit(cell);

        for(unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_value(phi_rho_tmp_2.get_value(q), q);

        phi.integrate_scatter(true, false, dst);
      }
    }
  }


  // Assemble Q0 rhs face term for the density update
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  assemble_rhs_face_term_rho_projection_Q0(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const std::vector<Vec>&                      src,
                                           const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEFaceEvaluation<dim, 0, n_q_points_1d, 1, Number>   phi_p(data, true, 2),
                                                           phi_m(data, false, 2),
                                                           phi_rho_old_p(data, true, 2),
                                                           phi_rho_old_m(data, false, 2);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_p.reinit(face);
        phi_m.reinit(face);

        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& point_vectorized = phi_p.quadrature_point(q);
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d)
              point[d] = point_vectorized[d][v];
            for(unsigned int d = 0; d < dim; ++d)
              velocity[d][v] = u.value(point, d);
          }

          const auto& n_plus       = phi_p.get_normal_vector(q);

          const auto& avg_flux     = 0.5*(phi_rho_old_p.get_value(q)*velocity +
                                          phi_rho_old_m.get_value(q)*velocity);
          const auto  lambda_old   = std::max(std::sqrt(scalar_product(velocity, velocity)),
                                              std::sqrt(scalar_product(velocity, velocity)));
          const auto& jump_rho_old = phi_rho_old_p.get_value(q) - phi_rho_old_m.get_value(q);

          phi_p.submit_value(-a21*dt*(scalar_product(avg_flux, n_plus) + 0.5*lambda_old*jump_rho_old), q);
          phi_m.submit_value(a21*dt*(scalar_product(avg_flux, n_plus) + 0.5*lambda_old*jump_rho_old), q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, 0, n_q_points_1d, 1, Number>   phi_p(data, true, 2),
                                                           phi_m(data, false, 2),
                                                           phi_rho_tmp_2_p(data, true, 2),
                                                           phi_rho_tmp_2_m(data, false, 2);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_tmp_2_p.reinit(face);
        phi_rho_tmp_2_p.gather_evaluate(src[0], true, false);
        phi_rho_tmp_2_m.reinit(face);
        phi_rho_tmp_2_m.gather_evaluate(src[0], true, false);
        phi_p.reinit(face);
        phi_m.reinit(face);

        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& point_vectorized = phi_p.quadrature_point(q);
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d)
              point[d] = point_vectorized[d][v];
            for(unsigned int d = 0; d < dim; ++d)
              velocity[d][v] = u.value(point, d);
          }

          const auto& n_plus         = phi_p.get_normal_vector(q);

          const auto& avg_flux_tmp_2 = 0.5*(phi_rho_tmp_2_p.get_value(q)*velocity +
                                            phi_rho_tmp_2_m.get_value(q)*velocity);
          const auto  lambda_tmp_2   = std::max(std::sqrt(scalar_product(velocity, velocity)),
                                                std::sqrt(scalar_product(velocity, velocity)));
          const auto& jump_rho_tmp_2 = phi_rho_tmp_2_p.get_value(q) - phi_rho_tmp_2_m.get_value(q);

          phi_p.submit_value(-a32*a21/a31*dt*(scalar_product(avg_flux_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_tmp_2), q);
          phi_m.submit_value(a32*a21/a31*dt*(scalar_product(avg_flux_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_tmp_2), q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
  }


  // Put together all the previous steps for density update
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  vmult_rhs_Q0(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d)
      src[d].update_ghost_values();

    this->data->loop(&HYPERBOLICOperator::assemble_rhs_cell_term_rho_projection_Q0,
                     &HYPERBOLICOperator::assemble_rhs_face_term_rho_projection_Q0,
                     &HYPERBOLICOperator::assemble_rhs_boundary_term_rho_projection_Q0,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble Q0 cell term for the density update
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  assemble_cell_term_rho_projection_Q0(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const Vec&                                   src,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, 0, n_q_points_1d, 1, Number> phi(data, 2);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, true, false);

      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(phi.get_value(q), q);

      phi.integrate_scatter(true, false, dst);
    }
  }


  // Put together all previous steps
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  apply_add(Vec& dst, const Vec& src) const {
    AssertIndexRange(NS_stage, 3);
    Assert(NS_stage > 0, ExcInternalError());

    if(NS_stage == 1) {
      this->data->cell_loop(&HYPERBOLICOperator::assemble_cell_term_rho_projection,
                            this, dst, src, false);
    }
    else {
      this->data->cell_loop(&HYPERBOLICOperator::assemble_cell_term_rho_projection_Q0,
                            this, dst, src, false);
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
    const double t_0;
    const double T;
    const double gamma;            //--- TR-BDF2 parameter
    const double a31;
    const double a32;
    unsigned int HYPERBOLIC_stage; //--- Flag to check at which current stage of TR-BDF2 are
    double       dt;

    Triangulation<dim> triangulation;

    FESystem<dim> fe_density;
    FESystem<dim> fe_velocity;
    FESystem<dim> fe_density_Q0;
    FESystem<dim> fe_velocity_Q0;

    DoFHandler<dim> dof_handler_density;
    DoFHandler<dim> dof_handler_velocity;
    DoFHandler<dim> dof_handler_density_Q0;
    DoFHandler<dim> dof_handler_velocity_Q0;

    QGaussLobatto<dim> quadrature_density;
    QGaussLobatto<dim> quadrature_velocity;

    QGauss<dim>        quadrature_accurate;

    LinearAlgebra::distributed::Vector<double> rho_old;
    LinearAlgebra::distributed::Vector<double> rho_tmp_2;
    LinearAlgebra::distributed::Vector<double> rho_curr;
    LinearAlgebra::distributed::Vector<double> rhs_rho;
    LinearAlgebra::distributed::Vector<double> rho_tmp;

    LinearAlgebra::distributed::Vector<double> u;

    LinearAlgebra::distributed::Vector<double> rho_old_Q0;
    LinearAlgebra::distributed::Vector<double> rho_tmp_2_Q0;
    LinearAlgebra::distributed::Vector<double> rho_curr_Q0;
    LinearAlgebra::distributed::Vector<double> rhs_rho_Q0;

    LinearAlgebra::distributed::Vector<double> u_Q0;

    DeclException2(ExcInvalidTimeStep,
                   double,
                   double,
                   << " The time step " << arg1 << " is out of range."
                   << std::endl
                   << " The permitted range is (0," << arg2 << "]");

    void create_triangulation(const unsigned int n_cells);

    void setup_dofs();

    void initialize();

    void update_density();

    void analyze_results();

    void output_results(const unsigned int step);

  private:
    EquationData::Density<dim>  rho_exact;
    EquationData::Velocity<dim> u_init;

    std::shared_ptr<MatrixFree<dim, double>> matrix_free_storage;

    HYPERBOLICOperator<dim, EquationData::degree, 2*EquationData::degree + 1,
                            LinearAlgebra::distributed::Vector<double>, double> euler_matrix;

    MGLevelObject<HYPERBOLICOperator<dim, EquationData::degree, 2*EquationData::degree + 1,
                                          LinearAlgebra::distributed::Vector<double>, double>> mg_matrices;

    AffineConstraints<double> constraints_velocity,
                              constraints_density,
                              constraints_density_Q0,
                              constraints_velocity_Q0;

    unsigned int vel_max_its;
    unsigned int vel_Krylov_size;
    unsigned int vel_off_diagonals;
    unsigned int vel_update_prec;
    double       vel_eps;
    double       vel_diag_strength;

    std::string saving_dir;

    ConditionalOStream pcout;

    std::ofstream      time_out;
    ConditionalOStream ptime_out;
    TimerOutput        time_table;

    std::ofstream output_rho;

    Vector<double> L2_error_per_cell_rho,
                   Linfty_error_per_cell_rho,
                   Linfty_error_per_cell_vel;

    double get_maximal_velocity();

    double get_minimal_density();

    double get_maximal_density();

    double get_minimal_density_Q0();

    double get_maximal_density_Q0();
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
    gamma(2.0 - std::sqrt(2.0)),     //--- Save also in the NavierStokes class the TR-BDF2 parameter value
    a31((2.0*gamma - 1.0)/6.0),
    a32((7.0 - 2.0*gamma)/6.0),
    HYPERBOLIC_stage(1),             //--- Initialize the flag for the TR_BDF2 stage
    dt(data.dt),
    triangulation(),
    fe_density(FE_DGQ<dim>(EquationData::degree), 1),
    fe_velocity(FE_DGQ<dim>(EquationData::degree), dim),
    fe_density_Q0(FE_DGQ<dim>(0), 1),
    fe_velocity_Q0(FE_DGQ<dim>(0), dim),
    dof_handler_density(triangulation),
    dof_handler_velocity(triangulation),
    dof_handler_density_Q0(triangulation),
    dof_handler_velocity_Q0(triangulation),
    quadrature_density(EquationData::degree + 1),
    quadrature_velocity(EquationData::degree + 1),
    quadrature_accurate(2*EquationData::degree + 1),
    rho_exact(data.initial_time),
    u_init(data.initial_time),
    euler_matrix(data),
    vel_max_its(data.vel_max_iterations),
    vel_Krylov_size(data.vel_Krylov_size),
    vel_off_diagonals(data.vel_off_diagonals),
    vel_update_prec(data.vel_update_prec),
    vel_eps(data.vel_eps),
    vel_diag_strength(data.vel_diag_strength),
    saving_dir(data.dir),
    pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    time_out("./" + data.dir + "/time_analysis_" +
             Utilities::int_to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)) + "proc.dat"),
    ptime_out(time_out, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    time_table(ptime_out, TimerOutput::summary, TimerOutput::cpu_and_wall_times),
    output_rho("./" + data.dir + "/error_analysis_rho.dat", std::ofstream::out) {
      AssertThrow(!((dt <= 0.0) || (dt > 0.5*T)), ExcInvalidTimeStep(dt, 0.5*T));

      matrix_free_storage = std::make_shared<MatrixFree<dim, double>>();

      constraints_velocity.clear();
      constraints_density.clear();
      constraints_density_Q0.clear();
      constraints_velocity_Q0.clear();

      create_triangulation(data.n_cells);
      setup_dofs();
      initialize();
  }


  // @sect4{<code>NavierStokesProjection::create_triangulation_and_dofs</code>}

  // The method that creates the triangulation.
  //
  template<int dim>
  void NavierStokesProjection<dim>::create_triangulation(const unsigned int n_cells) {
    TimerOutput::Scope t(time_table, "Create triangulation");

    GridGenerator::subdivided_hyper_cube(triangulation, n_cells, -1.0, 1.0, true);

    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> periodic_faces;
    GridTools::collect_periodic_faces(triangulation, 0, 1, 0, periodic_faces);
    triangulation.add_periodicity(periodic_faces);
  }


  // After creating the triangulation, it creates the mesh dependent
  // data, i.e. it distributes degrees of freedom and renumbers them, and
  // initializes the matrices and vectors that we will use.
  //
  template<int dim>
  void NavierStokesProjection<dim>::setup_dofs() {
    pcout << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl;
    pcout << "Number of levels: "       << triangulation.n_global_levels()       << std::endl;

    dof_handler_velocity.distribute_dofs(fe_velocity);
    dof_handler_density.distribute_dofs(fe_density);
    dof_handler_density_Q0.distribute_dofs(fe_density_Q0);
    dof_handler_velocity_Q0.distribute_dofs(fe_velocity_Q0);

    mg_matrices.clear_elements();
    dof_handler_velocity.distribute_mg_dofs();
    dof_handler_density.distribute_mg_dofs();
    dof_handler_density_Q0.distribute_mg_dofs();
    dof_handler_velocity_Q0.distribute_mg_dofs();

    pcout << "dim (V_h) = " << dof_handler_velocity.n_dofs()
          << std::endl
          << "dim (X_h) = " << dof_handler_density.n_dofs()
          << std::endl;
   pcout  << "CFL_u = " << dt*1.0*EquationData::degree*
                           std::sqrt(dim)/GridTools::minimal_cell_diameter(triangulation) << std::endl;

    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                            update_quadrature_points | update_values);
    additional_data.mapping_update_flags_inner_faces = (update_gradients | update_JxW_values | update_quadrature_points |
                                                        update_normal_vectors | update_values);
    additional_data.mapping_update_flags_boundary_faces = (update_gradients | update_JxW_values | update_quadrature_points |
                                                           update_normal_vectors | update_values);
    additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;

    std::vector<const DoFHandler<dim>*> dof_handlers;
    dof_handlers.push_back(&dof_handler_velocity);
    dof_handlers.push_back(&dof_handler_density);
    dof_handlers.push_back(&dof_handler_density_Q0);
    dof_handlers.push_back(&dof_handler_velocity_Q0);

    std::vector<const AffineConstraints<double>*> constraints;
    constraints.push_back(&constraints_velocity);
    constraints.push_back(&constraints_density);
    constraints.push_back(&constraints_density_Q0);
    constraints.push_back(&constraints_velocity_Q0);

    std::vector<QGauss<1>> quadratures;
    quadratures.push_back(QGauss<1>(2*EquationData::degree + 1));

    matrix_free_storage->reinit(dof_handlers, constraints, quadratures, additional_data);

    matrix_free_storage->initialize_dof_vector(u, 0);

    matrix_free_storage->initialize_dof_vector(rho_old, 1);
    matrix_free_storage->initialize_dof_vector(rho_tmp_2, 1);
    matrix_free_storage->initialize_dof_vector(rho_curr, 1);
    matrix_free_storage->initialize_dof_vector(rhs_rho, 1);
    matrix_free_storage->initialize_dof_vector(rho_tmp, 1);

    matrix_free_storage->initialize_dof_vector(rho_old_Q0, 2);
    matrix_free_storage->initialize_dof_vector(rho_tmp_2_Q0, 2);
    matrix_free_storage->initialize_dof_vector(rho_curr_Q0, 2);
    matrix_free_storage->initialize_dof_vector(rhs_rho_Q0, 2);

    matrix_free_storage->initialize_dof_vector(u_Q0, 3);

    Vector<double> error_per_cell_tmp(triangulation.n_active_cells());
    L2_error_per_cell_rho.reinit(error_per_cell_tmp);
    Linfty_error_per_cell_rho.reinit(error_per_cell_tmp);
    Linfty_error_per_cell_vel.reinit(error_per_cell_tmp);
  }


  // @sect4{ <code>NavierStokesProjection::initialize</code> }

  // This method loads the initial data
  //
  template<int dim>
  void NavierStokesProjection<dim>::initialize() {
    TimerOutput::Scope t(time_table, "Initialize state");

    VectorTools::interpolate(dof_handler_density, rho_exact, rho_old);
    VectorTools::interpolate(dof_handler_velocity, u_init, u);
    VectorTools::interpolate(dof_handler_density_Q0, rho_exact, rho_old_Q0);
    VectorTools::interpolate(dof_handler_velocity_Q0, u_init, u_Q0);
  }


  // @sect4{<code>NavierStokesProjection::update_density</code>}

  // This implements the update of the density for the hyperbolic part
  //
  template<int dim>
  void NavierStokesProjection<dim>::update_density() {
    TimerOutput::Scope t(time_table, "Update density");

    const std::vector<unsigned int> tmp = {1};
    const std::vector<unsigned int> tmp_Q0 = {2};

    if(HYPERBOLIC_stage == 1) {
      euler_matrix.initialize(matrix_free_storage, tmp, tmp);
      euler_matrix.vmult_rhs(rhs_rho, {rho_old});

      euler_matrix.initialize(matrix_free_storage, tmp_Q0, tmp_Q0);
      euler_matrix.vmult_rhs_Q0(rhs_rho_Q0, {rho_old_Q0});
    }
    else {
      euler_matrix.initialize(matrix_free_storage, tmp, tmp);
      euler_matrix.vmult_rhs(rhs_rho, {rho_tmp_2});

      euler_matrix.initialize(matrix_free_storage, tmp_Q0, tmp_Q0);
      euler_matrix.vmult_rhs_Q0(rhs_rho_Q0, {rho_tmp_2_Q0});
    }

    SolverControl solver_control(vel_max_its, vel_eps*rhs_rho.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

    SolverControl solver_control_Q0(vel_max_its, vel_eps*rhs_rho_Q0.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg_Q0(solver_control);

    if(HYPERBOLIC_stage == 1) {
      euler_matrix.set_NS_stage(1);
      euler_matrix.initialize(matrix_free_storage, tmp, tmp);
      rho_tmp_2.equ(1.0, rho_old);
      cg.solve(euler_matrix, rho_tmp_2, rhs_rho, PreconditionIdentity());

      euler_matrix.set_NS_stage(2);
      euler_matrix.initialize(matrix_free_storage, tmp_Q0, tmp_Q0);
      rho_tmp_2_Q0.equ(1.0, rho_old_Q0);
      cg_Q0.solve(euler_matrix, rho_tmp_2_Q0, rhs_rho_Q0, PreconditionIdentity());
      FETools::interpolate(dof_handler_density_Q0, rho_tmp_2_Q0, dof_handler_density, rho_tmp);

      rho_tmp_2.add(-1.0, rho_tmp);
      rho_tmp_2 /= (5.0*GridTools::minimal_cell_diameter(triangulation)/std::sqrt(dim)*gamma*dt);
      for(unsigned int i = 0; i < rho_tmp_2.local_size(); ++i)
        rho_tmp_2.local_element(i) *= (std::abs(rho_tmp_2.local_element(i)) <= 1.0);
      rho_tmp_2 *= (5.0*GridTools::minimal_cell_diameter(triangulation)/std::sqrt(dim)*gamma*dt);
      rho_tmp_2.add(1.0, rho_tmp);
    }
    else {
      euler_matrix.set_NS_stage(1);
      euler_matrix.initialize(matrix_free_storage, tmp, tmp);
      rho_curr.equ(1.0, rho_tmp_2);
      cg.solve(euler_matrix, rho_curr, rhs_rho, PreconditionIdentity());

      euler_matrix.set_NS_stage(2);
      euler_matrix.initialize(matrix_free_storage, tmp_Q0, tmp_Q0);
      rho_curr_Q0.equ(1.0, rho_tmp_2_Q0);
      cg_Q0.solve(euler_matrix, rho_curr_Q0, rhs_rho_Q0, PreconditionIdentity());
      FETools::interpolate(dof_handler_density_Q0, rho_curr_Q0, dof_handler_density, rho_tmp);
      rho_curr_Q0 *= a31/gamma;
      rho_curr_Q0.add(1.0 - a31/gamma, rho_old_Q0);

      rho_curr.add(-1.0, rho_tmp);
      rho_curr /= (5.0*GridTools::minimal_cell_diameter(triangulation)/std::sqrt(dim)*a32*gamma/a31*dt);
      for(unsigned int i = 0; i < rho_curr.local_size(); ++i)
        rho_curr.local_element(i) *= (std::abs(rho_curr.local_element(i)) <= 1.0);
      rho_curr *= (5.0*GridTools::minimal_cell_diameter(triangulation)/std::sqrt(dim)*a32*gamma/a31*dt);
      rho_curr.add(1.0, rho_tmp);
      rho_curr *= a31/gamma;
      rho_curr.add(1.0 - a31/gamma, rho_old);
    }
  }


  // Since we have solved a problem with analytic solution, we want to verify
  // the correctness of our implementation by computing the errors of the
  // numerical result against the analytic solution.
  //
  template <int dim>
  void NavierStokesProjection<dim>::analyze_results() {
    TimerOutput::Scope t(time_table, "Analysis results: computing errrors");

    rho_curr = 0;

    VectorTools::integrate_difference(dof_handler_density, rho_old, rho_exact,
                                      L2_error_per_cell_rho, quadrature_accurate, VectorTools::L2_norm);
    const double error_rho_L2 = VectorTools::compute_global_error(triangulation, L2_error_per_cell_rho, VectorTools::L2_norm);
    VectorTools::integrate_difference(dof_handler_density, rho_curr, rho_exact,
                                      L2_error_per_cell_rho, quadrature_accurate, VectorTools::L2_norm);
    const double L2_rho = VectorTools::compute_global_error(triangulation, L2_error_per_cell_rho, VectorTools::L2_norm);
    const double error_rel_rho_L2 = error_rho_L2/L2_rho;
    pcout << "Verification via L2 error density:    "<< error_rho_L2 << std::endl;
    pcout << "Verification via L2 relative error density:    "<< error_rel_rho_L2 << std::endl;

    VectorTools::integrate_difference(dof_handler_density, rho_old, rho_exact,
                                      Linfty_error_per_cell_rho, quadrature_accurate, VectorTools::Linfty_norm);
    const double error_rho_Linfty = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_rho, VectorTools::Linfty_norm);
    VectorTools::integrate_difference(dof_handler_density, rho_curr, rho_exact,
                                      Linfty_error_per_cell_rho, quadrature_accurate, VectorTools::Linfty_norm);
    const double Linfty_rho = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_rho, VectorTools::Linfty_norm);
    const double error_rel_rho_Linfty = error_rho_Linfty/Linfty_rho;
    pcout << "Verification via Linfty error density:    "<< error_rho_Linfty << std::endl;
    pcout << "Verification via Linfty relative error density:    "<< error_rel_rho_Linfty << std::endl;

    //const double ex_mean      = 0.5;
    const double ex_mean = 3.0/16.0;
    const double mean_value   = VectorTools::compute_mean_value(dof_handler_density, quadrature_accurate, rho_old, 0);
    //const double ex_variance  = std::sqrt(0.5);
    const double ex_variance = std::sqrt(26.0)/16.0;
    VectorTools::integrate_difference(dof_handler_density, rho_old, ConstantFunction<dim>(mean_value),
                                      L2_error_per_cell_rho, quadrature_accurate, VectorTools::L2_norm);
    const double variance     = VectorTools::compute_global_error(triangulation, L2_error_per_cell_rho, VectorTools::L2_norm)/
                                std::sqrt(2.0);
    const double dissipation  = (ex_variance - variance)*(ex_variance - variance) + (mean_value - ex_mean)*(mean_value - ex_mean);
    pcout << "Dissipation error:    "<< dissipation << std::endl;

    VectorTools::interpolate(dof_handler_density, rho_exact, rho_tmp_2);
    rho_tmp_2.add(-ex_mean);
    rho_curr.equ(1.0, rho_old);
    rho_curr.add(-mean_value);
    rho_tmp_2.scale(rho_curr);
    const double dispersion = 2.0*(ex_variance*variance -
                                   VectorTools::compute_mean_value(dof_handler_density, quadrature_accurate, rho_tmp_2, 0));
    pcout << "Dispersion error:    "<< dispersion << std::endl;

    //--- Analyze now Q0 results
    /*rho_curr_Q0 = 0;

    VectorTools::integrate_difference(dof_handler_density_Q0, rho_old_Q0, rho_exact,
                                      L2_error_per_cell_rho, quadrature_accurate, VectorTools::L2_norm);
    const double error_rho_L2_Q0 = VectorTools::compute_global_error(triangulation, L2_error_per_cell_rho, VectorTools::L2_norm);
    VectorTools::integrate_difference(dof_handler_density_Q0, rho_curr_Q0, rho_exact,
                                      L2_error_per_cell_rho, quadrature_accurate, VectorTools::L2_norm);
    const double L2_rho_Q0 = VectorTools::compute_global_error(triangulation, L2_error_per_cell_rho, VectorTools::L2_norm);
    const double error_rel_rho_L2_Q0 = error_rho_L2_Q0/L2_rho_Q0;
    pcout << "Verification via L2 error density Q0:    "<< error_rho_L2_Q0 << std::endl;
    pcout << "Verification via L2 relative error density Q0:    "<< error_rel_rho_L2_Q0 << std::endl;

    VectorTools::integrate_difference(dof_handler_density_Q0, rho_old_Q0, rho_exact,
                                      Linfty_error_per_cell_rho, quadrature_accurate, VectorTools::Linfty_norm);
    const double error_rho_Linfty_Q0 =
    VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_rho, VectorTools::Linfty_norm);
    VectorTools::integrate_difference(dof_handler_density_Q0, rho_curr_Q0, rho_exact,
                                      Linfty_error_per_cell_rho, quadrature_accurate, VectorTools::Linfty_norm);
    const double Linfty_rho_Q0 = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_rho, VectorTools::Linfty_norm);
    const double error_rel_rho_Linfty_Q0 = error_rho_Linfty_Q0/Linfty_rho_Q0;
    pcout << "Verification via Linfty error density Q0:    "<< error_rho_Linfty_Q0 << std::endl;
    pcout << "Verification via Linfty relative error density Q0:    "<< error_rel_rho_Linfty_Q0 << std::endl;

    const double mean_value_Q0   = VectorTools::compute_mean_value(dof_handler_density_Q0, quadrature_accurate, rho_old_Q0, 0);
    VectorTools::integrate_difference(dof_handler_density_Q0, rho_old_Q0, ConstantFunction<dim>(mean_value_Q0),
                                      L2_error_per_cell_rho, quadrature_accurate, VectorTools::L2_norm);
    const double variance_Q0     = VectorTools::compute_global_error(triangulation, L2_error_per_cell_rho, VectorTools::L2_norm)/
                                   std::sqrt(2.0);
    const double dissipation_Q0  = (ex_variance - variance_Q0)*(ex_variance - variance_Q0) +
                                   (mean_value_Q0 - ex_mean)*(mean_value_Q0 - ex_mean);
    pcout << "Dissipation error Q0:    "<< dissipation_Q0 << std::endl;

    VectorTools::interpolate(dof_handler_density_Q0, rho_exact, rho_tmp_2_Q0);
    rho_tmp_2_Q0.add(-ex_mean);
    rho_curr_Q0.equ(1.0, rho_old_Q0);
    rho_curr.add(-mean_value_Q0);
    rho_tmp_2_Q0.scale(rho_curr_Q0);
    const double dispersion_Q0 = 2.0*(ex_variance*variance_Q0 -
                                      VectorTools::compute_mean_value(dof_handler_density_Q0, quadrature_accurate, rho_tmp_2_Q0, 0));
    pcout << "Dispersion error Q0:    "<< dispersion_Q0 << std::endl;*/

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
      output_rho << error_rho_L2            << std::endl;
      output_rho << error_rel_rho_L2        << std::endl;
      output_rho << error_rho_Linfty        << std::endl;
      output_rho << error_rel_rho_Linfty    << std::endl;
      output_rho << dissipation             << std::endl;
      output_rho << dispersion              << std::endl;
      /*output_rho << error_rho_L2_Q0         << std::endl;
      output_rho << error_rel_rho_L2_Q0     << std::endl;
      output_rho << error_rho_Linfty_Q0     << std::endl;
      output_rho << error_rel_rho_Linfty_Q0 << std::endl;
      output_rho << dissipation_Q0          << std::endl;
      output_rho << dispersion_Q0           << std::endl;*/
    }
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

    rho_old.update_ghost_values();
    data_out.add_data_vector(dof_handler_density, rho_old, "rho", {DataComponentInterpretation::component_is_scalar});

    rho_old_Q0.update_ghost_values();
    data_out.add_data_vector(dof_handler_density_Q0, rho_old_Q0, "rho_Q0", {DataComponentInterpretation::component_is_scalar});

    data_out.build_patches();
    const std::string output = "./" + saving_dir + "/solution-" + Utilities::int_to_string(step, 5) + ".vtu";
    data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);
  }


  // The following function is used in determining the maximal velocity
  // in order to compute the CFL
  //
  template<int dim>
  double NavierStokesProjection<dim>::get_maximal_velocity() {
    VectorTools::integrate_difference(dof_handler_velocity, u, ZeroFunction<dim>(dim),
                                      Linfty_error_per_cell_vel, quadrature_velocity, VectorTools::Linfty_norm);
    const double res = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_vel, VectorTools::Linfty_norm);

    return res;
  }


  // The following function is used in determining the minimal density
  //
  template<int dim>
  double NavierStokesProjection<dim>::get_minimal_density() {
    FEValues<dim> fe_values(fe_density, quadrature_density, update_values);
    std::vector<double> solution_values(quadrature_density.size());

    double min_local_density = std::numeric_limits<double>::max();

    for(const auto& cell: dof_handler_density.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        fe_values.reinit(cell);
        fe_values.get_function_values(HYPERBOLIC_stage == 1 ? rho_tmp_2 : rho_curr, solution_values);
        for(unsigned int q = 0; q < quadrature_density.size(); ++q)
          min_local_density = std::min(min_local_density, solution_values[q]);
      }
    }

    return Utilities::MPI::min(min_local_density, MPI_COMM_WORLD);
  }


  // The following function is used in determining the minimal density
  //
  template<int dim>
  double NavierStokesProjection<dim>::get_maximal_density() {
    FEValues<dim> fe_values(fe_density, quadrature_density, update_values);
    std::vector<double> solution_values(quadrature_density.size());

    double max_local_density = std::numeric_limits<double>::min();

    for(const auto& cell: dof_handler_density.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        fe_values.reinit(cell);
        fe_values.get_function_values(HYPERBOLIC_stage == 1 ? rho_tmp_2 : rho_curr, solution_values);
        for(unsigned int q = 0; q < quadrature_density.size(); ++q)
          max_local_density = std::max(max_local_density, solution_values[q]);
      }
    }

    return Utilities::MPI::max(max_local_density, MPI_COMM_WORLD);
  }



  // The following function is used in determining the minimal density Q0
  //
  template<int dim>
  double NavierStokesProjection<dim>::get_minimal_density_Q0() {
    double min_local_density = std::numeric_limits<double>::max();

    for(const auto& cell: dof_handler_density_Q0.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        std::vector<types::global_dof_index> dof_indices(fe_density_Q0.dofs_per_cell);
        cell->get_dof_indices(dof_indices);
        min_local_density = std::min(min_local_density, HYPERBOLIC_stage == 1 ? rho_tmp_2_Q0(dof_indices[0]) :
                                                                                rho_curr_Q0(dof_indices[0]));
      }
    }

    return Utilities::MPI::min(min_local_density, MPI_COMM_WORLD);
  }


  // The following function is used in determining the minimal density
  //
  template<int dim>
  double NavierStokesProjection<dim>::get_maximal_density_Q0() {
    double max_local_density = std::numeric_limits<double>::min();

    for(const auto& cell: dof_handler_density_Q0.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        std::vector<types::global_dof_index> dof_indices(fe_density_Q0.dofs_per_cell);
        cell->get_dof_indices(dof_indices);
        max_local_density = std::max(max_local_density, HYPERBOLIC_stage == 1 ? rho_tmp_2_Q0(dof_indices[0]) :
                                                                                rho_curr_Q0(dof_indices[0]));
      }
    }

    return Utilities::MPI::max(max_local_density, MPI_COMM_WORLD);
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

    analyze_results();
    output_results(0);
    double time = t_0;
    unsigned int n = 0;
    while(std::abs(T - time) > 1e-10) {
      time += dt;
      n++;
      pcout << "Step = " << n << " Time = " << time << std::endl;

      //--- First stage of HYPERBOLIC operator
      euler_matrix.set_HYPERBOLIC_stage(HYPERBOLIC_stage);
      verbose_cout << "  Update density stage 1" << std::endl;
      update_density();
      pcout<<"Minimal density "<<get_minimal_density()<<std::endl;
      pcout<<"Maximal density "<<get_maximal_density()<<std::endl;
      /*pcout<<"Minimal density Q0 "<<get_minimal_density_Q0()<<std::endl;
      pcout<<"Maximal density Q0 "<<get_maximal_density_Q0()<<std::endl;*/
      HYPERBOLIC_stage = 2; //--- Flag to pass at second stage

      //--- Second stage of HYPERBOLIC operator
      euler_matrix.set_HYPERBOLIC_stage(HYPERBOLIC_stage);
      verbose_cout << "  Update density stage 2" << std::endl;
      update_density();
      const double rho_min = get_minimal_density();
      const double rho_max = get_maximal_density();
      /*const double rho_min_Q0 = get_minimal_density_Q0();
      const double rho_max_Q0 = get_maximal_density_Q0();*/
      pcout<<"Minimal density "<< rho_min <<std::endl;
      pcout<<"Maximal density "<< rho_max <<std::endl;
      /*pcout<<"Minimal density Q0 "<< rho_min_Q0 <<std::endl;
      pcout<<"Maximal density Q0 "<< rho_max_Q0 <<std::endl;*/
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
        output_rho << rho_min << std::endl;
        output_rho << rho_max << std::endl;
        /*output_rho << rho_min_Q0 << std::endl;
        output_rho << rho_max_Q0 << std::endl;*/
      }
      HYPERBOLIC_stage = 1; //--- Flag to pass at first stage at next step

      //--- Update for next step
      rho_old.equ(1.0, rho_curr);
      rho_old_Q0.equ(1.0, rho_curr_Q0);
      rho_exact.advance_time(dt);
      analyze_results();
      if(n % output_interval == 0) {
        verbose_cout << "Plotting Solution final" << std::endl;
        output_results(n);
      }
      if(T - time < dt && T - time > 1e-10) {
        dt = T - time;
        euler_matrix.set_dt(dt);
      }
    }
    if(n % output_interval != 0) {
      verbose_cout << "Plotting Solution final" << std::endl;
      output_results(n);
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

    NavierStokesProjection<1> test(data);
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

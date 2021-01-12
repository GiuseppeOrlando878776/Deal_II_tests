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

      double dt;
      double initial_time;
      double final_time;

      double Reynolds;

      unsigned int n_global_refines;

      unsigned int vel_max_iterations;
      unsigned int vel_Krylov_size;
      unsigned int vel_off_diagonals;
      unsigned int vel_update_prec;
      double       vel_eps;
      double       vel_diag_strength;

      bool         verbose;
      unsigned int output_interval;

    protected:
      ParameterHandler prm;
    };

    // In the constructor of this class we declare all the parameters.
    Data_Storage::Data_Storage(): dt(5e-4),
                                  initial_time(0.0),
                                  final_time(1.0),
                                  Reynolds(1.0),
                                  n_global_refines(0),
                                  vel_max_iterations(1000),
                                  vel_Krylov_size(30),
                                  vel_off_diagonals(60),
                                  vel_update_prec(15),
                                  vel_eps(1e-12),
                                  vel_diag_strength(0.01),
                                  verbose(true),
                                  output_interval(15) {
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
        prm.declare_entry("n_of_refines",
                          "0",
                          Patterns::Integer(0, 15),
                          " The number of global refines we do on the mesh. ");
      }
      prm.leave_subsection();

      prm.enter_subsection("Data solve velocity");
      {
        prm.declare_entry("max_iterations",
                          "1000",
                          Patterns::Integer(1, 10000),
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
        n_global_refines = prm.get_integer("n_of_refines");
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
      if(component == 0) {
        const double Um = 1.5;
        const double H  = 4.1;
        return 4.0*Um*p(1)*(H - p(1))/(H * H);
      }
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
    double Pressure<dim>::value(const Point<dim>&  p,
                                const unsigned int component) const {
      (void)component;
      AssertIndexRange(component, 1);
      return 25.0 - p(0);
    }

  } // namespace EquationData



  // @sect3{ <code>NavierStokesProjectionOperator::NavierStokesProjectionOperator</code> }
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  class NavierStokesProjectionOperator: public MatrixFreeOperators::Base<dim, Vec> {
  public:
    NavierStokesProjectionOperator();

    NavierStokesProjectionOperator(RunTimeParameters::Data_Storage& data);

    void set_TR_BDF2_stage(const unsigned int stage);

    void set_NS_stage(const unsigned int stage);

    void set_u_extr(const Vec& src);

    void set_vel_exact(const EquationData::Velocity<dim>& src);

    void vmult_rhs_velocity(Vec& dst, const std::vector<Vec>& src) const;

    void vmult_rhs_pressure(Vec& dst, const Vec& src) const;

    virtual void compute_diagonal() override;

  protected:
    RunTimeParameters::Data_Storage& eq_data;

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
    const double theta = 1.0;

    const double a21 = 0.5;
    const double a22 = 0.5;
    const double C_p = 5.0;
    const double C_u = 0.1;

    Vec                         u_extr;
    EquationData::Velocity<dim> vel_exact;

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
                                         const Vec&                                   src,
                                         const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const Vec&                                   src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                             Vec&                                         dst,
                                             const Vec&                                   src,
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
                                         const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_diagonal_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const Vec&                                   src,
                                              const std::pair<unsigned int, unsigned int>& cell_range) const;

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
                                                  const std::pair<unsigned int, unsigned int>& face_range) const;
  };


  // Default constructor
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::NavierStokesProjectionOperator():
    MatrixFreeOperators::Base<dim, Vec>(), Re(), dt(), gamma(), a31(), a32(), a33(), TR_BDF2_stage(), NS_stage(1), u_extr() {}


  // Constructor with runtime parameters storage
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  NavierStokesProjectionOperator(RunTimeParameters::Data_Storage& data):
    MatrixFreeOperators::Base<dim, Vec>(), eq_data(data), Re(data.Reynolds), dt(data.dt),
                                           gamma(2.0 - std::sqrt(2.0)), a31((1.0 - gamma)/(2.0*(2.0 - gamma))),
                                           a32(a31), a33(1.0/(2.0 - gamma)), TR_BDF2_stage(1), NS_stage(1), u_extr() {}


  // Setter of TR-BDF2 stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  set_TR_BDF2_stage(const unsigned int stage) {
    Assert(stage == 1 || stage == 2, ExcInternalError());
    TR_BDF2_stage = stage;
  }


  // Setter of NS stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  set_NS_stage(const unsigned int stage) {
    Assert(stage == 1 || stage == 2, ExcInternalError());
    NS_stage = stage;
  }


  // Setter of extrapolated velocity for different stages
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  set_u_extr(const Vec& src) {
    u_extr = src;
  }


  // Setter of exact velocity for boundary
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  set_vel_exact(const EquationData::Velocity<dim>& src) {
    vel_exact = src;
  }


  // Assemble rhs cell term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_rhs_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(TR_BDF2_stage == 1) {
      FEEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi(data, 0), phi_old(data, 0), phi_old_extr(data, 0);
      FEEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi_old_press(data, 1);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_old.reinit(cell);
        phi_old.gather_evaluate(src[0], true, true);
        phi_old_extr.reinit(cell);
        phi_old_extr.gather_evaluate(src[1], true, false); //Questo sarà u_(n + gamma/2)
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
          phi.submit_gradient(-a21/Re*grad_u_n - a21*tensor_product_u_n + p_n_times_identity, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi(data, 0), phi_old(data, 0), phi_int(data, 0);
      FEEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi_old_press(data, 1);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_old.reinit(cell);
        phi_old.gather_evaluate(src[0], true, true);
        phi_int.reinit(cell);
        phi_int.gather_evaluate(src[1], true, true); //Questo sarà u_(n + gamma)
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
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_rhs_face_term_velocity(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi_p(data, true, 0), phi_m(data, false, 0),
                                                                     phi_old_p(data, true, 0), phi_old_m(data, false, 0),
                                                                     phi_old_extr_p(data, true, 0), phi_old_extr_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi_old_press_p(data, true, 1), phi_old_press_m(data, false, 1);

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
          phi_p.submit_value((a21/Re*avg_grad_u_old + a21*avg_tensor_product_u_n)*n_plus - avg_p_old*n_plus, q);
          phi_m.submit_value(-(a21/Re*avg_grad_u_old + a21*avg_tensor_product_u_n)*n_plus + avg_p_old*n_plus, q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi_p(data, true, 0), phi_m(data, false, 0),
                                                                     phi_old_p(data, true, 0), phi_old_m(data, false, 0),
                                                                     phi_int_p(data, true, 0), phi_int_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi_old_press_p(data, true, 1), phi_old_press_m(data, false, 1);

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
                              a31*avg_tensor_product_u_n - a32*avg_tensor_product_u_n_gamma)*n_plus -
                              avg_p_old*n_plus, q);
          phi_m.submit_value(-(a31/Re*avg_grad_u_old + a32/Re*avg_grad_u_int -
                               a31*avg_tensor_product_u_n - a32*avg_tensor_product_u_n_gamma)*n_plus +
                               avg_p_old*n_plus, q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
  }


  // Assemble rhs boundary term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_rhs_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                      Vec&                                         dst,
                                      const std::vector<Vec>&                      src,
                                      const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi(data, true, 0),
                                                                     phi_old(data, true, 0),
                                                                     phi_old_extr(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi_old_press(data, true, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        const auto boundary_id = data.get_boundary_id(face);
        phi_old.reinit(face);
        phi_old.gather_evaluate(src[0], true, true);
        phi_old_extr.reinit(face);
        phi_old_extr.gather_evaluate(src[1], true, false);
        phi_old_press.reinit(face);
        phi_old_press.gather_evaluate(src[2], true, false);
        phi.reinit(face);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus             = phi.get_normal_vector(q);
          const auto& grad_u_old         = phi_old.get_gradient(q);
          const auto& tensor_product_u_n = outer_product(phi_old.get_value(q), phi_old_extr.get_value(q));
          const auto& p_old              = phi_old_press.get_value(q);
          switch(boundary_id) {
            case 1:
            {
              phi.submit_value((a21/Re*grad_u_old + a21*tensor_product_u_n)*n_plus - p_old*n_plus, q);
              break;
            }
            case 2:
            {
              Vector<double> boundary_values(dim);
              const auto& p_vectorized = phi.quadrature_point(q);
              Point<dim> p;
              for(unsigned d = 0; d < dim; ++d)
                p[d] = p_vectorized[d][0];
              vel_exact.vector_value(p, boundary_values);
              Tensor<1, dim, VectorizedArray<Number>> u_int_m;
              for(unsigned d = 0; d < dim; ++d)
                u_int_m[d] = boundary_values[d];
              const auto   tensor_product_u_int_m = outer_product(u_int_m, phi_old_extr.get_value(q));
              const auto   lambda                 = std::abs(scalar_product(phi_old_extr.get_value(q), n_plus));
              const auto&  face_iterator          = data.get_face_iterator(face, 0);
              const double measure                = face_iterator.first->face(face_iterator.second)->measure();
              phi.submit_value((a21/Re*grad_u_old + a21*tensor_product_u_n)*n_plus - p_old*n_plus +
                               a22/Re*C_u/measure*u_int_m -
                               0.5*a22*tensor_product_u_int_m*n_plus +
                               0.5*a22*lambda*u_int_m, q);
              phi.submit_gradient(-a22/Re*outer_product(u_int_m, n_plus), q);
              break;
            }
            case 3:
            {
              phi.submit_value((a21/Re*grad_u_old + a21*tensor_product_u_n)*n_plus - p_old*n_plus, q);
              break;
            }
            case 4:
            {
              phi.submit_value((a21/Re*grad_u_old + a21*tensor_product_u_n)*n_plus - p_old*n_plus, q);
              break;
            }
            default:
              Assert(false, ExcNotImplemented());
          }
        }
        if(boundary_id == 2)
          phi.integrate_scatter(true, true, dst);
        else
          phi.integrate_scatter(true, false, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi(data, true, 0),
                                                                     phi_old(data, true, 0),
                                                                     phi_int(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi_old_press(data, true, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++ face) {
        const auto boundary_id = data.get_boundary_id(face);
        phi_old.reinit(face);
        phi_old.gather_evaluate(src[0], true, true);
        phi_int.reinit(face);
        phi_int.gather_evaluate(src[1], true, true);
        phi_old_press.reinit(face);
        phi_old_press.gather_evaluate(src[2], true, false);
        phi.reinit(face);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus                   = phi.get_normal_vector(q);
          const auto& grad_u_old               = phi_old.get_gradient(q);
          const auto& grad_u_int               = phi_int.get_gradient(q);
          const auto& tensor_product_u_n       = outer_product(phi_old.get_value(q), phi_old.get_value(q));
          const auto& tensor_product_u_n_gamma = outer_product(phi_int.get_value(q), phi_int.get_value(q));
          const auto& p_old                    = phi_old_press.get_value(q);
          switch(boundary_id) {
            case 1:
            {
              phi.submit_value((a31/Re*grad_u_old + a32/Re*grad_u_int -
                                a31*tensor_product_u_n - a32*tensor_product_u_n_gamma)*n_plus - p_old*n_plus, q);
              break;
            }
            case 2:
            {
              Vector<double> boundary_values(dim);
              const auto& p_vectorized = phi.quadrature_point(q);
              Point<dim> p;
              for(unsigned d = 0; d < dim; ++d)
                p[d] = p_vectorized[d][0];
              vel_exact.vector_value(p, boundary_values);
              Tensor<1, dim, VectorizedArray<Number>> u_m;
              for(unsigned d = 0; d < dim; ++d)
                u_m[d] = boundary_values[d];
              FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi_extr(data, true, 0);
              phi_extr.reinit(face);
              phi_extr.gather_evaluate(src[3], true, false);
              const auto   tensor_product_u_m = outer_product(u_m, phi_extr.get_value(q));
              const auto   lambda             = std::abs(scalar_product(phi_extr.get_value(q), n_plus));
              const auto&  face_iterator      = data.get_face_iterator(face, 0);
              const double measure            = face_iterator.first->face(face_iterator.second)->measure();
              phi.submit_value((a31/Re*grad_u_old + a32/Re*grad_u_int -
                                a31*tensor_product_u_n - a32*tensor_product_u_n_gamma)*n_plus - p_old*n_plus +
                               a33/Re*C_u/measure*u_m - 0.5*a33*tensor_product_u_m*n_plus + 0.5*a33*lambda*u_m, q);
              phi.submit_gradient(-a33/Re*outer_product(u_m, n_plus), q);
              break;
            }
            case 3:
            {
              phi.submit_value((a31/Re*grad_u_old + a32/Re*grad_u_int -
                                a31*tensor_product_u_n - a32*tensor_product_u_n_gamma)*n_plus - p_old*n_plus, q);
              break;
            }
            case 4:
            {
              phi.submit_value((a31/Re*grad_u_old + a32/Re*grad_u_int -
                                a31*tensor_product_u_n - a32*tensor_product_u_n_gamma)*n_plus - p_old*n_plus, q);
              break;
            }
            default:
              Assert(false, ExcNotImplemented());
          }
        }
        if(boundary_id == 2)
          phi.integrate_scatter(true, true, dst);
        else
          phi.integrate_scatter(true, false, dst);
      }
    }
  }


  // Put together all the previous steps for velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  vmult_rhs_velocity(Vec& dst, const std::vector<Vec>& src) const {
    this->data->loop(&NavierStokesProjectionOperator::assemble_rhs_cell_term_velocity,
                     &NavierStokesProjectionOperator::assemble_rhs_face_term_velocity,
                     &NavierStokesProjectionOperator::assemble_rhs_boundary_term_velocity,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble rhs cell term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_rhs_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const Vec&                                   src,
                                  const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi(data, 1);
    FEEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi_proj(data, 0);

    const double coeff = (TR_BDF2_stage == 1) ? 1.0/(gamma*dt) : 1.0/((1.0 - gamma)*dt);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_proj.reinit(cell);
      phi_proj.gather_evaluate(src, true, false);
      phi.reinit(cell);
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& u_star_star = phi_proj.get_value(q);
        phi.submit_gradient(-coeff*u_star_star, q);
      }
      phi.integrate_scatter(false, true, dst);
    }
  }


  // Assemble rhs face term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_rhs_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const Vec&                                   src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi_p(data, true, 1), phi_m(data, false, 1);
    FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi_proj_p(data, true, 0), phi_proj_m(data, false, 0);

    const double coeff = (TR_BDF2_stage == 1) ? 1.0/(gamma*dt) : 1.0/((1.0 - gamma)*dt);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_proj_p.reinit(face);
      phi_proj_p.gather_evaluate(src, true, false);
      phi_proj_m.reinit(face);
      phi_proj_m.gather_evaluate(src, true, false);
      phi_p.reinit(face);
      phi_m.reinit(face);
      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus           = phi_p.get_normal_vector(q);
        const auto& avg_u_star_star  = 0.5*(phi_proj_p.get_value(q) + phi_proj_m.get_value(q));
        phi_p.submit_value(coeff*scalar_product(avg_u_star_star, n_plus), q);
        phi_m.submit_value(-coeff*scalar_product(avg_u_star_star, n_plus), q);
      }
      phi_p.integrate_scatter(true, false, dst);
      phi_m.integrate_scatter(true, false, dst);
    }
  }


  // Assemble rhs boundary term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_rhs_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                      Vec&                                         dst,
                                      const Vec&                                   src,
                                      const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi(data, true, 1);
    FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi_proj(data, true, 0);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_proj.reinit(face);
      phi_proj.gather_evaluate(src, true, false);
      phi.reinit(face);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(scalar_product(phi_proj.get_value(q), phi.get_normal_vector(q)), q);
      phi.integrate_scatter(true, false, dst);
    }
  }


  // Put together all the previous steps for pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  vmult_rhs_pressure(Vec& dst, const Vec& src) const {
    this->data->loop(&NavierStokesProjectionOperator::assemble_rhs_cell_term_pressure,
                     &NavierStokesProjectionOperator::assemble_rhs_face_term_pressure,
                     &NavierStokesProjectionOperator::assemble_rhs_boundary_term_pressure,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble cell term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(TR_BDF2_stage == 1) {
      FEEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi(data, 0), phi_old_extr(data, 0);

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
      FEEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi(data, 0), phi_int_extr(data, 0);

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
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_face_term_velocity(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi_p(data, true, 0), phi_m(data, false, 0),
                                                                     phi_old_extr_p(data, true, 0), phi_old_extr_m(data, false, 0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        const auto& face_iterator = data.get_face_iterator(face, 0);
        const double measure      = face_iterator.first->face(face_iterator.second)->measure();
        const double coef_jump    = C_u/measure;
        phi_p.reinit(face);
        phi_p.gather_evaluate(src, true, true);
        phi_m.reinit(face);
        phi_m.gather_evaluate(src, true, true);
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
                             a22*avg_tensor_product_u_int*n_plus + 0.5*a22*lambda*jump_u_int , q);
          phi_m.submit_value(-a22/Re*(-avg_grad_u_int*n_plus + coef_jump*jump_u_int) -
                              a22*avg_tensor_product_u_int*n_plus - 0.5*a22*lambda*jump_u_int, q);
          phi_p.submit_gradient(-theta*0.5*a22/Re*outer_product(jump_u_int, n_plus), q);
          phi_m.submit_gradient(-theta*0.5*a22/Re*outer_product(jump_u_int, n_plus), q);
        }
        phi_p.integrate_scatter(true, true, dst);
        phi_m.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi_p(data, true, 0), phi_m(data, false, 0),
                                                                     phi_extr_p(data, true, 0), phi_extr_m(data, false, 0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        const auto& face_iterator = data.get_face_iterator(face, 0);
        const double measure      = face_iterator.first->face(face_iterator.second)->measure();
        const double coef_jump    = C_u/measure;
        phi_p.reinit(face);
        phi_p.gather_evaluate(src, true, true);
        phi_m.reinit(face);
        phi_m.gather_evaluate(src, true, true);
        phi_extr_p.reinit(face);
        phi_extr_p.gather_evaluate(u_extr, true, false);
        phi_extr_m.reinit(face);
        phi_extr_m.gather_evaluate(u_extr, true, false);
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
          phi_p.submit_gradient(-theta*0.5*a33/Re*outer_product(jump_u, n_plus), q);
          phi_m.submit_gradient(-theta*0.5*a33/Re*outer_product(jump_u, n_plus), q);
        }
        phi_p.integrate_scatter(true, true, dst);
        phi_m.integrate_scatter(true, true, dst);
      }
    }
  }


  // Assemble boundary term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const Vec&                                   src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi(data, true, 0), phi_old_extr(data, true, 0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        const auto boundary_id    = data.get_boundary_id(face);
        const auto& face_iterator = data.get_face_iterator(face, 0);
        const double measure      = face_iterator.first->face(face_iterator.second)->measure();
        const double coef_trasp   = (boundary_id == 1 || boundary_id == 4) ? 0.0 : 0.5;
        const double coef_jump    = C_u/measure;
        phi.reinit(face);
        phi.gather_evaluate(src, true, true);
        phi_old_extr.reinit(face);
        phi_old_extr.gather_evaluate(u_extr, true, false);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus               = phi.get_normal_vector(q);
          const auto& tensor_product_n     = outer_product(n_plus, n_plus);
          const auto& grad_u_int           = phi.get_gradient(q);
          const auto& u_int                = phi.get_value(q);
          const auto& tensor_product_u_int = outer_product(phi.get_value(q), phi_old_extr.get_value(q));
          const auto  lambda               = std::abs(scalar_product(phi_old_extr.get_value(q), n_plus));
          const auto  coef_lambda          = (boundary_id == 1 || boundary_id == 4) ? lambda : 0.5*lambda;
          switch(boundary_id) {
            case 1:
            {
              phi.submit_value(a22/Re*(-grad_u_int*n_plus + coef_jump*u_int) +
                               a22*coef_trasp*tensor_product_u_int*n_plus + a22*coef_lambda*u_int, q);
              phi.submit_gradient(-theta*a22/Re*outer_product(u_int, n_plus), q);
              break;
            }
            case 2:
            {
              phi.submit_value(a22/Re*(-grad_u_int*n_plus + coef_jump*u_int) +
                               a22*coef_trasp*tensor_product_u_int*n_plus + a22*coef_lambda*u_int, q);
              phi.submit_gradient(-theta*a22/Re*outer_product(u_int, n_plus), q);
              break;
            }
            case 3:
            {
              const auto u_int_m                  = u_int - 2.0*scalar_product(u_int, n_plus)*n_plus;
              const auto tensor_product_u_int_m   = outer_product(u_int_m, phi_old_extr.get_value(q));
              const auto avg_tensor_product_u_int = 0.5*(tensor_product_u_int + tensor_product_u_int_m);
              phi.submit_value(a22/Re*(-grad_u_int*n_plus + coef_jump*(u_int - u_int_m)) +
                               a22*avg_tensor_product_u_int*n_plus + a22*coef_lambda*(u_int - u_int_m), q);
              phi.submit_gradient(-theta*a22/Re*outer_product(u_int - u_int_m, n_plus), q);
              break;
            }
            case 4:
            {
              phi.submit_value(a22/Re*(-grad_u_int*n_plus + coef_jump*u_int) +
                               a22*coef_trasp*tensor_product_u_int*n_plus + a22*coef_lambda*u_int, q);
              phi.submit_gradient(-theta*a22/Re*outer_product(u_int, n_plus), q);
              break;
            }
            default:
              Assert(false, ExcInternalError());
          }
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi(data, true, 0), phi_extr(data, true, 0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        const auto& face_iterator = data.get_face_iterator(face, 0);
        const double measure = face_iterator.first->face(face_iterator.second)->measure();
        const auto boundary_id = data.get_boundary_id(face);
        const double coef_trasp = (boundary_id == 1 || boundary_id == 4) ? 0.0 : 0.5;
        const double coef_jump  = C_u/measure;
        phi.reinit(face);
        phi.gather_evaluate(src, true, true);
        phi_extr.reinit(face);
        phi_extr.gather_evaluate(u_extr, true, false);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus           = phi.get_normal_vector(q);
          const auto& tensor_product_n = outer_product(n_plus, n_plus);
          const auto& grad_u           = phi.get_gradient(q);
          const auto& u                = phi.get_value(q);
          const auto& tensor_product_u = outer_product(phi.get_value(q), phi_extr.get_value(q));
          const auto  lambda           = std::abs(scalar_product(phi_extr.get_value(q), n_plus));
          const auto  coef_lambda      = (boundary_id == 1 || boundary_id == 4) ? lambda : 0.5*lambda;
          switch(boundary_id) {
            case 1:
            {
              phi.submit_value(a33/Re*(-grad_u*n_plus + coef_jump*u) +
                               coef_trasp*a33*tensor_product_u*n_plus + a33*coef_lambda*u, q);
              phi.submit_gradient(-theta*a33/Re*outer_product(u, n_plus), q);
              break;
            }
            case 2:
            {
              phi.submit_value(a33/Re*(-grad_u*n_plus + coef_jump*u) +
                               coef_trasp*a33*tensor_product_u*n_plus + a33*coef_lambda*u, q);
              phi.submit_gradient(-theta*a33/Re*outer_product(u, n_plus), q);
              break;
            }
            case 3:
            {
              const auto u_m                  = u - 2.0*scalar_product(u, n_plus)*n_plus;
              const auto tensor_product_u_m   = outer_product(u_m, phi_extr.get_value(q));
              const auto avg_tensor_product_u = 0.5*(tensor_product_u + tensor_product_u_m);
              phi.submit_value(a33/Re*(-grad_u*n_plus + coef_jump*(u - u_m)) +
                               a33*avg_tensor_product_u*n_plus + a33*coef_lambda*(u - u_m), q);
              phi.submit_gradient(-theta*a33/Re*outer_product(u - u_m, n_plus), q);
              break;
            }
            case 4:
            {
              phi.submit_value(a33/Re*(-grad_u*n_plus + coef_jump*u) +
                               coef_trasp*a33*tensor_product_u*n_plus + a33*coef_lambda*u, q);
              phi.submit_gradient(-theta*a33/Re*outer_product(u, n_plus), q);
              break;
            }
            default:
              Assert(false, ExcInternalError());
          }
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
  }


  // Assemble cell term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number> phi(data, 1);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, false, true);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_gradient(-phi.get_gradient(q), q);
      phi.integrate_scatter(false, true, dst);
    }
  }


  // Assemble face term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_face_term_pressure(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number> phi_p(data, true, 1), phi_m(data, false, 1);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      const auto&  face_iterator = data.get_face_iterator(face, 0);
      const double measure       = face_iterator.first->face(face_iterator.second)->measure();
      const double coef_jump     = C_p/measure;
      phi_p.reinit(face);
      phi_p.gather_evaluate(src, true, true);
      phi_m.reinit(face);
      phi_m.gather_evaluate(src, true, true);
      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus        = phi_p.get_normal_vector(q);
        const auto& avg_grad_pres = 0.5*(phi_p.get_gradient(q) + phi_m.get_gradient(q));
        const auto& jump_pres     = phi_p.get_value(q) - phi_m.get_value(q);
        phi_p.submit_value(scalar_product(avg_grad_pres, n_plus) - coef_jump*jump_pres, q);
        phi_m.submit_value(-scalar_product(avg_grad_pres, n_plus) + coef_jump*jump_pres, q);
        phi_p.submit_gradient(theta*0.5*jump_pres*n_plus, q);
        phi_m.submit_gradient(theta*0.5*jump_pres*n_plus, q);
      }
      phi_p.integrate_scatter(true, true, dst);
      phi_m.integrate_scatter(true, true, dst);
    }
  }


  // Assemble boundary term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const Vec&                                   src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number> phi(data, true, 1);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      const auto&  face_iterator = data.get_face_iterator(face, 0);
      const double measure       = face_iterator.first->face(face_iterator.second)->measure();
      const double coef_jump     = C_p/measure;
      phi.reinit(face);
      phi.gather_evaluate(src, true, false);
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& n_plus = phi.get_normal_vector(q);
        const auto& pres   = phi.get_value(q);
        phi.submit_value(-coef_jump*pres, q);
      }
      phi.integrate_scatter(true, false, dst);
    }
  }


  // Assemble boundary term for the pressure
  //
  /*template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const Vec&                                   src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number> phi(data, true, 1);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi.reinit(face);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(0.0, q);
      phi.integrate_scatter(true, false, dst);
    }
  }*/


  // Put together all previous steps
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
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
    else
      Assert(false, ExcNotImplemented());
  }


  // Assemble diagonal cell term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_diagonal_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const Vec&                                   src,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(TR_BDF2_stage == 1) {
      FEEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi(data, 0), phi_old_extr(data, 0);

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
      FEEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi(data, 0), phi_int_extr(data, 0);

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


  // Assemble diagonal cell term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_diagonal_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const Vec&                                   src,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number> phi(data, 1);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
          phi.submit_dof_value(VectorizedArray<Number>(), j);
        phi.submit_dof_value(make_vectorized_array<Number>(1.0), i);

        phi.evaluate(false, true);
        for(unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_gradient(-phi.get_gradient(q), q);
        phi.integrate(false, true);
        diagonal[i] = phi.get_dof_value(i);
      }
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i)
        phi.submit_dof_value(diagonal[i], i);
      phi.distribute_local_to_global(dst);
    }
  }


  // Assemble diagonal face term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_diagonal_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const Vec&                                   src,
                                       const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number> phi_p(data, true, 1), phi_m(data, false, 1);

    AssertDimension(phi_p.dofs_per_component, phi_m.dofs_per_component);
    AlignedVector<VectorizedArray<Number>> diagonal(phi_p.dofs_per_component);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      const auto&  face_iterator = data.get_face_iterator(face, 0);
      const double measure       = face_iterator.first->face(face_iterator.second)->measure();
      const double coef_jump     = C_p/measure;
      phi_p.reinit(face);
      phi_m.reinit(face);
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
          phi_p.submit_value(scalar_product(avg_grad_pres, n_plus) - coef_jump*jump_pres, q);
          phi_m.submit_value(-scalar_product(avg_grad_pres, n_plus) + coef_jump*jump_pres, q);
          phi_p.submit_gradient(theta*0.5*jump_pres*n_plus, q);
          phi_m.submit_gradient(theta*0.5*jump_pres*n_plus, q);
        }
        phi_p.integrate(true, true);
        phi_m.integrate(true, true);
        diagonal[i] = phi_p.get_dof_value(i) + phi_m.get_dof_value(i);
      }
      for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i)
        phi_p.submit_dof_value(diagonal[i], i);
      phi_p.distribute_local_to_global(dst);
    }
  }


  // Assemble boundary term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_diagonal_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const Vec&                                   src,
                                           const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number> phi(data, true, 1);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      const auto&  face_iterator = data.get_face_iterator(face, 0);
      const double measure       = face_iterator.first->face(face_iterator.second)->measure();
      const double coef_jump     = C_p/measure;
      phi.reinit(face);
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
          phi.submit_dof_value(VectorizedArray<Number>(), j);
        phi.submit_dof_value(make_vectorized_array<Number>(1.0), i);

        phi.evaluate(true, false);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus = phi.get_normal_vector(q);
          const auto& pres   = phi.get_value(q);
          phi.submit_value(-coef_jump*pres, q);
        }
        phi.integrate(true, false);
        diagonal[i] = phi.get_dof_value(i);
      }
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i)
        phi.submit_dof_value(diagonal[i], i);
      phi.distribute_local_to_global(dst);
    }
  }


  // Put together all previous steps
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::compute_diagonal() {
    if(NS_stage == 1) {
      this->inverse_diagonal_entries.reset(new DiagonalMatrix<Vec>());
      auto& inverse_diagonal = this->inverse_diagonal_entries->get_vector();
      this->data->initialize_dof_vector(inverse_diagonal, 0);
      Vec dummy;
      dummy.reinit(inverse_diagonal.local_size());
      this->data->cell_loop(&NavierStokesProjectionOperator::assemble_diagonal_cell_term_velocity,
                            this, inverse_diagonal, dummy, false);
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
    else
      Assert(false, ExcNotImplemented());
  }



  // @sect3{The <code>NavierStokesProjection</code> class}

  // Now for the main class of the program. It implements the various versions
  // of the projection method for Navier-Stokes equations.
  //
  template<int dim>
  class NavierStokesProjection {
  public:
    NavierStokesProjection(RunTimeParameters::Data_Storage& data);

    void run(const bool verbose = false, const unsigned int n_plots = 10);

  protected:
    const double       dt;
    const double       t_0;
    const double       T;
    const double       gamma;         //--- TR-BDF2 parameter
    unsigned int       TR_BDF2_stage; //--- Flag to check at which current stage of TR-BDF2 are
    const double       Re;

    EquationData::Velocity<dim>               vel_exact;
    std::map<types::global_dof_index, double> boundary_values;
    std::vector<types::boundary_id>           boundary_ids;

    Triangulation<dim> triangulation;

    FESystem<dim> fe_velocity;
    FESystem<dim> fe_pressure;

    DoFHandler<dim> dof_handler_velocity;
    DoFHandler<dim> dof_handler_pressure;

    QGauss<dim> quadrature_pressure;
    QGauss<dim> quadrature_velocity;

    SparseMatrix<double> pres_Diff;

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

    DeclException2(ExcInvalidTimeStep,
                   double,
                   double,
                   << " The time step " << arg1 << " is out of range."
                   << std::endl
                   << " The permitted range is (0," << arg2 << "]");

    void create_triangulation_and_dofs(const unsigned int n_refines);

    void initialize();

    void interpolate_velocity();

    void diffusion_step();

    void projection_step();

  private:
    std::shared_ptr<MatrixFree<dim, double>> matrix_free_storage;

    NavierStokesProjectionOperator<dim,
                                   EquationData::degree_p,
                                   EquationData::degree_p + 1,
                                   EquationData::degree_p + 2,
                                   LinearAlgebra::distributed::Vector<double>,
                                   double> navier_stokes_matrix;

    AffineConstraints<double> constraints,
                              constraints_velocity, constraints_pressure;

    SparsityPattern  sparsity_pattern_velocity,
                     sparsity_pattern_pres_vel;

    unsigned int vel_max_its;
    unsigned int vel_Krylov_size;
    unsigned int vel_off_diagonals;
    unsigned int vel_update_prec;
    double       vel_eps;
    double       vel_diag_strength;

    // The next few structures and functions are for doing various things in
    // parallel.
    //
    // One of the things that are specific to this program is that we don't just
    // have a single DoFHandler object that represents both the velocities and
    // the pressure, but we use individual DoFHandler objects for these two
    // kinds of variables. We pay for this optimization when we want to assemble
    // terms that involve both variables, such as the divergence of the velocity
    // and the gradient of the pressure, times the respective test functions.
    // When doing so, we can't just anymore use a single FEValues object, but
    // rather we need two, and they need to be initialized with cell iterators
    // that point to the same cell in the triangulation but different
    // DoFHandlers.
    //
    // To do this in practice, we declare a "synchronous" iterator -- an object
    // that internally consists of several (in our case two) iterators, and each
    // time the synchronous iteration is moved forward one step, each of the
    // iterators stored internally is moved forward one step as well, thereby
    // always staying in sync. As it so happens, there is a deal.II class that
    // facilitates this sort of thing. (What is important here is to know that
    // two DoFHandler objects built on the same triangulation will walk over the
    // cells of the triangulation in the same order.)
    using IteratorTuple = std::tuple<typename DoFHandler<dim>::active_cell_iterator,
                                     typename DoFHandler<dim>::active_cell_iterator>;
    using IteratorPair = SynchronousIterators<IteratorTuple>;

    struct InitGradPerTaskData {
      unsigned int                         vel_dpc;
      unsigned int                         pres_dpc;
      FullMatrix<double>                   local_grad;
      std::vector<types::global_dof_index> vel_local_dof_indices;
      std::vector<types::global_dof_index> pres_local_dof_indices;

      InitGradPerTaskData(const unsigned int vdpc,
                          const unsigned int pdpc): vel_dpc(vdpc),
                                                    pres_dpc(pdpc),
                                                    local_grad(vdpc, pdpc),
                                                    vel_local_dof_indices(vdpc),
                                                    pres_local_dof_indices(pdpc) {}
    };

    struct InitGradScratchData {
      unsigned int  nqp;
      FEValues<dim> fe_val_vel;
      FEValues<dim> fe_val_pres;
      InitGradScratchData(const FESystem<dim>& fe_v, const FESystem<dim>& fe_p,
                          const QGauss<dim>& quad, const UpdateFlags flags_v,
                          const UpdateFlags flags_p): nqp(quad.size()),
                                                      fe_val_vel(fe_v, quad, flags_v),
                                                      fe_val_pres(fe_p, quad, flags_p) {}
      InitGradScratchData(const InitGradScratchData& data): nqp(data.nqp),
                                                            fe_val_vel(data.fe_val_vel.get_fe(),
                                                                       data.fe_val_vel.get_quadrature(),
                                                                       data.fe_val_vel.get_update_flags()),
                                                            fe_val_pres(data.fe_val_pres.get_fe(),
                                                                        data.fe_val_pres.get_quadrature(),
                                                                        data.fe_val_pres.get_update_flags()) {}
    };

    void apply_momentum_bcs();

    void assemble_one_cell_of_gradient(const IteratorPair&  SI,
                                       InitGradScratchData& scratch,
                                       InitGradPerTaskData& data);

    void copy_gradient_local_to_global(const InitGradPerTaskData& data);

    void initialize_gradient_operator();

    // The final function implements postprocessing the output
    void output_results(const unsigned int step);
  };


  // @sect4{ <code>NavierStokesProjection::NavierStokesProjection</code> }

  // In the constructor, we just read all the data from the
  // <code>Data_Storage</code> object that is passed as an argument, verify that
  // the data we read are reasonable and, finally, create the triangulation and
  // load the initial data.
  template<int dim>
  NavierStokesProjection<dim>::NavierStokesProjection(RunTimeParameters::Data_Storage& data):
    dt(data.dt),
    t_0(data.initial_time),
    T(data.final_time),
    gamma(2.0 - std::sqrt(2.0)), //--- Save in the NavierStokes class the TR-BDF2 parameter value
    TR_BDF2_stage(1),  //--- Initialize the flag for the TR_BDF2 stage
    Re(data.Reynolds),
    vel_exact(data.initial_time),
    fe_velocity(FE_DGQ<dim>(EquationData::degree_p + 1), dim),
    fe_pressure(FE_DGQ<dim>(EquationData::degree_p), 1),
    dof_handler_velocity(triangulation),
    dof_handler_pressure(triangulation),
    quadrature_pressure(EquationData::degree_p + 2),
    quadrature_velocity(EquationData::degree_p + 2),
    navier_stokes_matrix(data),
    vel_max_its(data.vel_max_iterations),
    vel_Krylov_size(data.vel_Krylov_size),
    vel_off_diagonals(data.vel_off_diagonals),
    vel_update_prec(data.vel_update_prec),
    vel_eps(data.vel_eps),
    vel_diag_strength(data.vel_diag_strength) {
      if(EquationData::degree_p < 1) {
        std::cout
        << " WARNING: The chosen pair of finite element spaces is not stable."
        << std::endl
        << " The obtained results will be nonsense" << std::endl;
      }

      AssertThrow(!((dt <= 0.0) || (dt > 0.5*T)), ExcInvalidTimeStep(dt, 0.5*T));

      constraints.clear();
      create_triangulation_and_dofs(data.n_global_refines);
      initialize();
  }


  // @sect4{<code>NavierStokesProjection::create_triangulation_and_dofs</code>}

  // The method that creates the triangulation and refines it the needed number
  // of times. After creating the triangulation, it creates the mesh dependent
  // data, i.e. it distributes degrees of freedom and renumbers them, and
  // initializes the matrices and vectors that we will use.
  //
  template<int dim>
  void NavierStokesProjection<dim>::create_triangulation_and_dofs(const unsigned int n_refines) {
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);

    std::string   filename = "nsbench2.inp";
    std::ifstream file(filename);
    Assert(file, ExcFileNotOpen(filename.c_str()));
    grid_in.read_ucd(file);

    std::cout << "Number of refines = " << n_refines << std::endl;
    triangulation.refine_global(n_refines);
    std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;

    dof_handler_velocity.distribute_dofs(fe_velocity);
    dof_handler_pressure.distribute_dofs(fe_pressure);

    boundary_ids = triangulation.get_boundary_ids();

    initialize_gradient_operator();

    std::cout << "dim (X_h) = " << dof_handler_velocity.n_dofs()
              << std::endl
              << "dim (M_h) = " << dof_handler_pressure.n_dofs()
              << std::endl
              << "Re        = " << Re << std::endl
              << std::endl;

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
    dof_handlers.push_back(&dof_handler_pressure);
    std::vector<const AffineConstraints<double>*> constraints;
    constraints.push_back(&constraints_velocity);
    constraints.push_back(&constraints_pressure);
    matrix_free_storage = std::make_unique<MatrixFree<dim, double>>();
    matrix_free_storage->reinit(dof_handlers, constraints, QGauss<dim - 1>(EquationData::degree_p + 2), additional_data);
    matrix_free_storage->initialize_dof_vector(u_star, 0);
    matrix_free_storage->initialize_dof_vector(rhs_u, 0);
    matrix_free_storage->initialize_dof_vector(u_n, 0);
    matrix_free_storage->initialize_dof_vector(u_extr, 0);
    matrix_free_storage->initialize_dof_vector(u_n_minus_1, 0);
    matrix_free_storage->initialize_dof_vector(u_n_gamma, 0);
    matrix_free_storage->initialize_dof_vector(u_tmp, 0);
    matrix_free_storage->initialize_dof_vector(pres_int, 1);
    matrix_free_storage->initialize_dof_vector(pres_n, 1);
    matrix_free_storage->initialize_dof_vector(rhs_p, 1);

    DynamicSparsityPattern dsp(dof_handler_velocity.n_dofs(), dof_handler_velocity.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_velocity, dsp, constraints_velocity, false);
    sparsity_pattern_velocity.copy_from(dsp);
  }


  // @sect4{ <code>NavierStokesProjection::initialize</code> }

  // This method loads the initial data
  //
  template<int dim>
  void NavierStokesProjection<dim>::initialize() {
    EquationData::Pressure<dim> pres(t_0);
    pres.advance_time(dt);
    VectorTools::interpolate(dof_handler_pressure, pres, pres_n);

    vel_exact.set_time(t_0);
    VectorTools::interpolate(dof_handler_velocity, vel_exact, u_n_minus_1);
    vel_exact.advance_time(dt);
    VectorTools::interpolate(dof_handler_velocity, vel_exact, u_n);
  }


  template<int dim>
  void NavierStokesProjection<dim>::apply_momentum_bcs() {
    constraints_velocity.clear();

    for(const auto& boundary_id : boundary_ids) {
      switch(boundary_id) {
        case 1:
          VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                   boundary_id,
                                                   Functions::ZeroFunction<dim>(2),
                                                   constraints_velocity);
          break;
        case 2:
          VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                   boundary_id,
                                                   vel_exact,
                                                   constraints_velocity);
          break;
        case 3:
          {
            std::vector<bool> selector = {true, false};
            VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                     boundary_id,
                                                     Functions::ZeroFunction<dim>(2),
                                                     constraints_velocity,
                                                     ComponentMask(selector));
            break;
          }
        case 4:
          VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                   boundary_id,
                                                   Functions::ZeroFunction<dim>(2),
                                                   constraints_velocity);
          break;
        default:
          Assert(false, ExcNotImplemented());
      }
    }

    constraints_velocity.close();
  }


  // For the gradient operator, it is important to notice that the gradient
  // operator acts from the pressure space into the velocity space, so we have
  // to deal with two different finite element spaces. To keep the loops
  // synchronized, we use the alias that we have defined before, namely
  // <code>PairedIterators</code> and <code>IteratorPair</code>.
  template<int dim>
  void NavierStokesProjection<dim>::assemble_one_cell_of_gradient(const IteratorPair&  SI,
                                                                  InitGradScratchData& scratch,
                                                                  InitGradPerTaskData& data) {
    scratch.fe_val_vel.reinit(std::get<0>(*SI));
    scratch.fe_val_pres.reinit(std::get<1>(*SI));

    std::get<0>(*SI)->get_dof_indices(data.vel_local_dof_indices);
    std::get<1>(*SI)->get_dof_indices(data.pres_local_dof_indices);

    data.local_grad = 0;
    for(unsigned int q = 0; q < scratch.nqp; ++q) {
      for(unsigned int i = 0; i < data.vel_dpc; ++i) {
        const unsigned d = scratch.fe_val_vel.get_fe().system_to_component_index(i).first;
        for(unsigned int j = 0; j < data.pres_dpc; ++j) {
          data.local_grad(i, j) += -scratch.fe_val_vel.shape_grad(i, q)[d] *
                                    scratch.fe_val_pres.shape_value(j, q) *
                                    scratch.fe_val_vel.JxW(q);
        }
      }
    }
  }


  template<int dim>
  void NavierStokesProjection<dim>::copy_gradient_local_to_global(const InitGradPerTaskData& data) {
    constraints.distribute_local_to_global(data.local_grad, data.vel_local_dof_indices,
                                           data.pres_local_dof_indices, pres_Diff);
    pres_Diff.compress(VectorOperation::add);
  }


  template<int dim>
  void NavierStokesProjection<dim>::initialize_gradient_operator() {
    DynamicSparsityPattern dsp(dof_handler_velocity.n_dofs(), dof_handler_pressure.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_velocity, dof_handler_pressure, dsp);
    sparsity_pattern_pres_vel.copy_from(dsp);
    pres_Diff.reinit(sparsity_pattern_pres_vel);

    InitGradPerTaskData per_task_data(fe_velocity.n_dofs_per_cell(), fe_pressure.n_dofs_per_cell());
    UpdateFlags flags_v = update_gradients | update_JxW_values;
    UpdateFlags flags_p = update_values;
    InitGradScratchData scratch_data(fe_velocity, fe_pressure, quadrature_velocity, flags_v, flags_p);

    WorkStream::run(IteratorPair(IteratorTuple(dof_handler_velocity.begin_active(),
                                               dof_handler_pressure.begin_active())),
                    IteratorPair(IteratorTuple(dof_handler_velocity.end(),
                                               dof_handler_pressure.end())),
                    *this,
                    &NavierStokesProjection<dim>::assemble_one_cell_of_gradient,
                    &NavierStokesProjection<dim>::copy_gradient_local_to_global,
                    scratch_data,
                    per_task_data);
  }


  // @sect4{<code>NavierStokesProjection::interpolate_velocity</code>}

  // This function computes the extrapolated velocity to be used in the momentum predictor
  //
  template<int dim>
  void NavierStokesProjection<dim>::interpolate_velocity() {
    //--- TR-BDF2 first step
    if(TR_BDF2_stage == 1) {
      u_extr.equ(1.0 + gamma/(2.0*(1.0 - gamma)), u_n);
      u_tmp.equ(gamma/(2.0*(1.0 - gamma)), u_n_minus_1);
      u_extr -= u_tmp;
    }
    //--- TR-BDF2 second step
    else {
      u_extr.equ(1.0 + (1.0 - gamma)/gamma, u_star);
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
    const std::vector<unsigned int> tmp = {0};
    navier_stokes_matrix.initialize(matrix_free_storage, tmp, tmp);
    navier_stokes_matrix.set_vel_exact(vel_exact);
    if(TR_BDF2_stage == 1)
      navier_stokes_matrix.vmult_rhs_velocity(rhs_u, {u_n, u_extr, pres_n});
    else
      navier_stokes_matrix.vmult_rhs_velocity(rhs_u, {u_n, u_n_gamma, pres_n, u_extr});

    navier_stokes_matrix.set_NS_stage(1);
    navier_stokes_matrix.set_u_extr(u_extr);

    SolverControl solver_control(vel_max_its, vel_eps*rhs_u.l2_norm());
    SolverGMRES<LinearAlgebra::distributed::Vector<double>>
    gmres(solver_control, SolverGMRES<LinearAlgebra::distributed::Vector<double>>::AdditionalData(vel_Krylov_size));
    PreconditionJacobi<NavierStokesProjectionOperator<dim,
                                                      EquationData::degree_p,
                                                      EquationData::degree_p + 1,
                                                      EquationData::degree_p + 2,
                                                      LinearAlgebra::distributed::Vector<double>,
                                                      double>> preconditioner;
    navier_stokes_matrix.compute_diagonal();
    preconditioner.initialize(navier_stokes_matrix);
    gmres.solve(navier_stokes_matrix, u_star, rhs_u, preconditioner);
  }


  // @sect4{<code>NavierStokesProjection::projection_step</code>}

  // This implements the projection step
  //
  template<int dim>
  void NavierStokesProjection<dim>::projection_step() {
    const std::vector<unsigned int> tmp = {1};
    navier_stokes_matrix.initialize(matrix_free_storage, tmp, tmp);
    navier_stokes_matrix.vmult_rhs_pressure(rhs_p, u_star);

    navier_stokes_matrix.set_NS_stage(2);

    SolverControl solver_control(vel_max_its, vel_eps*rhs_p.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
    if(TR_BDF2_stage == 1)
      cg.solve(navier_stokes_matrix, pres_int, rhs_p, PreconditionIdentity());
    else
      cg.solve(navier_stokes_matrix, pres_n, rhs_p, PreconditionIdentity());
  }


  // @sect4{ <code>NavierStokesProjection::output_results</code> }

  // This method plots the current solution. The main difficulty is that we want
  // to create a single output file that contains the data for all velocity
  // components, the pressure, and also the vorticity of the flow. On the other
  // hand, velocities and the pressure live on separate DoFHandler objects, and
  // so can't be written to the same file using a single DataOut object. As a
  // consequence, we have to work a bit harder to get the various pieces of data
  // into a single DoFHandler object, and then use that to drive graphical
  // output.
  //
  // We will not elaborate on this process here, but rather refer to step-32,
  // where a similar procedure is used (and is documented) to create a joint
  // DoFHandler object for all variables.
  //
  // Let us also note that we here compute the vorticity as a scalar quantity in
  // a separate function, using the $L^2$ projection of the quantity
  // $\text{curl} u$ onto the finite element space used for the components of
  // the velocity. In principle, however, we could also have computed as a
  // pointwise quantity from the velocity, and do so through the
  // DataPostprocessor mechanism discussed in step-29 and step-33.
  template<int dim>
  void NavierStokesProjection<dim>::output_results(const unsigned int step) {
    const FESystem<dim> joint_fe(fe_velocity, 1, fe_pressure, 1);
    DoFHandler<dim>     joint_dof_handler(triangulation);
    joint_dof_handler.distribute_dofs(joint_fe);
    Assert(joint_dof_handler.n_dofs() == (dof_handler_velocity.n_dofs() + dof_handler_pressure.n_dofs()),
           ExcInternalError());
    Vector<double> joint_solution(joint_dof_handler.n_dofs());
    std::vector<types::global_dof_index> loc_joint_dof_indices(joint_fe.n_dofs_per_cell()),
                                         loc_vel_dof_indices(fe_velocity.n_dofs_per_cell()),
                                         loc_pres_dof_indices(fe_pressure.n_dofs_per_cell());
    typename DoFHandler<dim>::active_cell_iterator joint_beginc = joint_dof_handler.begin_active(),
                                                   joint_endc   = joint_dof_handler.end(),
                                                   vel_cell     = dof_handler_velocity.begin_active(),
                                                   pres_cell    = dof_handler_pressure.begin_active();
    for(auto joint_cell = joint_beginc; joint_cell != joint_endc; ++joint_cell, ++vel_cell, ++pres_cell) {
      joint_cell->get_dof_indices(loc_joint_dof_indices);
      vel_cell->get_dof_indices(loc_vel_dof_indices);
      pres_cell->get_dof_indices(loc_pres_dof_indices);
      for(unsigned int i = 0; i < joint_fe.n_dofs_per_cell(); ++i) {
        switch(joint_fe.system_to_base_index(i).first.first) {
          case 0:
            Assert(joint_fe.system_to_base_index(i).first.second < dim,
                   ExcInternalError());
            joint_solution(loc_joint_dof_indices[i]) =
            u_n(loc_vel_dof_indices[joint_fe.system_to_base_index(i).second]);
            break;
          case 1:
            Assert(joint_fe.system_to_base_index(i).first.second == 0,
                   ExcInternalError());
            joint_solution(loc_joint_dof_indices[i]) = pres_n(loc_pres_dof_indices[joint_fe.system_to_base_index(i).second]);
            break;
          default:
            Assert(false, ExcInternalError());
        }
      }
    }
    std::vector<std::string> joint_solution_names(dim, "v");
    joint_solution_names.emplace_back("p");
    DataOut<dim> data_out;
    data_out.attach_dof_handler(joint_dof_handler);
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(dim + 1, DataComponentInterpretation::component_is_part_of_vector);
    component_interpretation[dim] = DataComponentInterpretation::component_is_scalar;
    data_out.add_data_vector(joint_solution, joint_solution_names, DataOut<dim>::type_dof_data, component_interpretation);
    data_out.build_patches(EquationData::degree_p + 1);
    std::ofstream output("solution-" + Utilities::int_to_string(step, 5) + ".vtk");
    data_out.write_vtk(output);
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
  void NavierStokesProjection<dim>::run(const bool         verbose,
                                        const unsigned int output_interval) {
    ConditionalOStream verbose_cout(std::cout, verbose);

    const auto n_steps = static_cast<unsigned int>((T - t_0) / dt);
    output_results(1);
    for(unsigned int n = 2; n <= n_steps; ++n) {
      if(n % output_interval == 0) {
        verbose_cout << "Plotting Solution" << std::endl;
        output_results(n);
      }
      std::cout << "Step = " << n << " Time = " << (n * dt) << std::endl;
      //--- First stage of TR-BDF2
      verbose_cout << "  Interpolating the velocity stage 1" << std::endl;
      interpolate_velocity();
      verbose_cout << "  Diffusion Step stage 1 " << std::endl;
      navier_stokes_matrix.set_TR_BDF2_stage(TR_BDF2_stage);
      vel_exact.advance_time(gamma*dt);
      u_star = u_extr;
      diffusion_step();
      verbose_cout << "  Projection Step stage 1" << std::endl;
      pres_Diff.vmult(u_tmp, pres_n);
      u_tmp.equ(gamma*dt, u_tmp);
      u_star += u_tmp;
      projection_step();
      verbose_cout << "  Updating the Velocity stage 1" << std::endl;
      u_n_gamma.equ(1.0, u_star);
      pres_Diff.vmult(u_tmp, pres_int);
      u_tmp.equ(-gamma*dt, u_tmp);
      u_n_gamma += u_tmp;
      TR_BDF2_stage = 2; //--- Flag to pass at second stage
      //--- Second stage of TR-BDF2
      u_n_minus_1 = u_n;
      verbose_cout << "  Interpolating the velocity stage 2" << std::endl;
      interpolate_velocity();
      verbose_cout << "  Diffusion Step stage 2 " << std::endl;
      navier_stokes_matrix.set_TR_BDF2_stage(TR_BDF2_stage);
      vel_exact.advance_time((1.0 - gamma)*dt);
      u_star = u_extr;
      diffusion_step();
      verbose_cout << "  Projection Step stage 2" << std::endl;
      pres_Diff.vmult(u_tmp, pres_int);
      u_tmp.equ((1.0 - gamma)*dt, u_tmp);
      u_star += u_tmp;
      projection_step();
      verbose_cout << "  Updating the Velocity stage 2" << std::endl;
      u_n.equ(1.0, u_star);
      pres_Diff.vmult(u_tmp, pres_n);
      u_tmp.equ((gamma - 1.0)*dt, u_tmp);
      u_n += u_tmp;
      TR_BDF2_stage = 1; //--- Flag to pass at first stage at next step
    }
    output_results(n_steps);
  }

} // namespace Step35


// @sect3{ The main function }

// The main function looks very much like in all the other tutorial programs, so
// there is little to comment on here:
int main() {
  try {
    using namespace Step35;

    RunTimeParameters::Data_Storage data;
    data.read_data("parameter-file.prm");

    deallog.depth_console(data.verbose ? 2 : 0);

    NavierStokesProjection<2> test(data);
    test.run(data.verbose, data.output_interval);
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
  std::cout << "----------------------------------------------------"
            << std::endl
            << "Apparently everything went fine!" << std::endl
            << "Don't forget to brush your teeth :-)" << std::endl
            << std::endl;
  return 0;
}

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

#include <random>

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
      double CFL;

      unsigned int n_global_refines;

      unsigned int vel_max_iterations;
      unsigned int vel_Krylov_size;
      unsigned int vel_off_diagonals;
      unsigned int vel_update_prec;
      double       vel_eps;
      double       vel_diag_strength;

      bool         verbose;
      unsigned int output_interval;

      std::string dir;

    protected:
      ParameterHandler prm;
    };

    // In the constructor of this class we declare all the parameters.
    Data_Storage::Data_Storage(): initial_time(0.0),
                                  final_time(1.0),
                                  Reynolds(1.0),
                                  CFL(1.0),
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
        prm.declare_entry("CFL",
                          "1.0",
                          Patterns::Double(0.0),
                          " The Courant number. ");
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
        CFL = prm.get_double("CFL");
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

      dir = prm.get("saving directory");

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

      Velocity(const double Re, const double initial_time = 0.0);

      virtual double value(const Point<dim>&  p,
                           const unsigned int component = 0) const override;

      virtual void vector_value(const Point<dim>& p,
                                Vector<double>&   values) const override;

      virtual Tensor<1, dim, double> gradient(const Point<dim>& p,
                                              const unsigned int component = 0) const override;

      virtual void vector_gradient(const Point<dim>& p, std::vector<Tensor<1, dim, double>>& gradients) const override;

    private:
      double Re;
    };


    template<int dim>
    Velocity<dim>::Velocity(const double initial_time): Function<dim>(dim, initial_time), Re() {}


    template<int dim>
    Velocity<dim>::Velocity(const double Re, const double initial_time): Function<dim>(dim, initial_time), Re(Re) {}


    template<int dim>
    double Velocity<dim>::value(const Point<dim>& p, const unsigned int component) const {
      AssertIndexRange(component, 2);
      const double curr_time = this->get_time();
      if(component == 0)
        return std::cos(p(0))*std::sin(p(1))*std::exp(-2.0*curr_time/Re);
      else
        return -std::sin(p(0))*std::cos(p(1))*std::exp(-2.0*curr_time/Re);
    }


    template<int dim>
    void Velocity<dim>::vector_value(const Point<dim>& p, Vector<double>& values) const {
      Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));
      for(unsigned int i = 0; i < dim; ++i)
        values[i] = value(p, i);
    }


    template<int dim>
    Tensor<1, dim, double> Velocity<dim>::gradient(const Point<dim>& p, const unsigned int component) const {
      AssertIndexRange(component, 2);
      Tensor<1, dim, double> result;
      const double curr_time = this->get_time();
      if(component == 0) {
        result[0] = -std::sin(p(0))*std::sin(p(1))*std::exp(-2.0*curr_time/Re);
        result[1] =  std::cos(p(0))*std::cos(p(1))*std::exp(-2.0*curr_time/Re);
      }
      else {
        result[0] = -std::cos(p(0))*std::cos(p(1))*std::exp(-2.0*curr_time/Re);
        result[1] =  std::sin(p(0))*std::sin(p(1))*std::exp(-2.0*curr_time/Re);
      }
      return result;
    }


    template<int dim>
    void Velocity<dim>::vector_gradient(const Point<dim>& p, std::vector<Tensor<1, dim, double>>& gradients) const {
      Assert(gradients.size() == dim, ExcDimensionMismatch(gradients.size(), dim));
      for(unsigned int i = 0; i < dim; ++i)
        gradients[i] = gradient(p, i);
    }


    // We do the same for the pressure (since it is a scalar field) we can derive
    // directly from the deal.II built-in class Function
    template<int dim>
    class Pressure: public Function<dim> {
    public:
      Pressure(const double initial_time = 0.0);

      Pressure(const double Re, const double initial_time = 0.0);

      virtual double value(const Point<dim>&  p,
                           const unsigned int component = 0) const override;

    private:
      double Re;
    };


    template<int dim>
    Pressure<dim>::Pressure(const double initial_time): Function<dim>(1, initial_time), Re() {}


    template<int dim>
    Pressure<dim>::Pressure(const double Re, const double initial_time): Function<dim>(1, initial_time), Re(Re) {}


    template<int dim>
    double Pressure<dim>::value(const Point<dim>&  p, const unsigned int component) const {
      (void)component;
      AssertIndexRange(component, 1);
      const double curr_time = this->get_time();
      return -0.25*(std::cos(2.0*p(0)) + std::cos(2.0*p(1)))*std::exp(-4.0*curr_time/Re);
    }

  } // namespace EquationData



  // @sect3{ <code>NavierStokesProjectionOperator::NavierStokesProjectionOperator</code> }
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  class NavierStokesProjectionOperator: public MatrixFreeOperators::Base<dim, Vec> {
  public:
    NavierStokesProjectionOperator();

    NavierStokesProjectionOperator(RunTimeParameters::Data_Storage& data);

    void set_dt(const double time_step);

    void set_TR_BDF2_stage(const unsigned int stage);

    void set_NS_stage(const unsigned int stage);

    void set_u_extr(const Vec& src);

    void set_u_n(const Vec& src);

    void set_time(const double curr_time);

    void advance_p_time(const double time_adv);

    void vmult_rhs_velocity(Vec& dst, const std::vector<Vec>& src) const;

    void vmult_rhs_pressure(Vec& dst, const std::vector<Vec>& src) const;

    void vmult_grad_p_projection(Vec& dst, const Vec& src) const;

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
    const double a21 = 0.5;
    const double a22 = 0.5;

    const double theta_v = 1.0;
    const double theta_p = 1.0;
    const double C_p = 1.0*(fe_degree_p + 1)*(fe_degree_p + 1);
    const double C_u = 1.0*(fe_degree_v + 1)*(fe_degree_v + 1);

    Vec                         u_extr;
    Vec                         u_n;

    EquationData::Velocity<dim> vel_exact;
    EquationData::Pressure<dim> pres_exact;

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
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::NavierStokesProjectionOperator():
    MatrixFreeOperators::Base<dim, Vec>(), Re(), dt(), gamma(), a31(), a32(), a33(), TR_BDF2_stage(), NS_stage(1), u_extr() {}


  // Constructor with runtime parameters storage
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  NavierStokesProjectionOperator(RunTimeParameters::Data_Storage& data):
    MatrixFreeOperators::Base<dim, Vec>(), eq_data(data), Re(data.Reynolds), dt(5e-4),
                                           gamma(1.0), a31((1.0 - gamma)/(2.0*(2.0 - gamma))),
                                           a32(a31), a33(1.0/(2.0 - gamma)), TR_BDF2_stage(1), NS_stage(1), u_extr(),
                                           vel_exact(data.Reynolds, data.initial_time),
                                           pres_exact(data.Reynolds, data.initial_time) {}

  // Setter of time-step
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  set_dt(const double time_step) {
    dt = time_step;
  }


  // Advance time for gradient of pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  advance_p_time(const double time_adv) {
    pres_exact.advance_time(time_adv);
  }


  // Setter of TR-BDF2 stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  set_TR_BDF2_stage(const unsigned int stage) {
    AssertIndexRange(stage, 3);
    Assert(stage > 0, ExcInternalError());
    TR_BDF2_stage = stage;
  }


  // Setter of NS stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  set_NS_stage(const unsigned int stage) {
    AssertIndexRange(stage, 4);
    Assert(stage > 0, ExcInternalError());
    NS_stage = stage;
  }


  // Setter of extrapolated velocity for different stages
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  set_u_extr(const Vec& src) {
    u_extr = src;
    u_extr.update_ghost_values();
  }


  // Setter of old velocity for different stages
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  set_u_n(const Vec& src) {
    u_n = src;
    u_n.update_ghost_values();
  }


  // Setter of exact velocity for boundary conditions
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::set_time(const double curr_time) {
    vel_exact.set_time(curr_time);
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
      FEEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi(data, 0), phi_old(data, 0);
      FEEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number> phi_old_press(data, 1);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_old.reinit(cell);
        phi_old.gather_evaluate(src[0], true, true);
        phi_old_press.reinit(cell);
        phi_old_press.gather_evaluate(src[2], true, false);
        phi.reinit(cell);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& u_n                = phi_old.get_value(q);
          const auto& grad_u_n           = phi_old.get_gradient(q);
          const auto& tensor_product_u_n = outer_product(u_n, u_n);
          const auto& p_n                = phi_old_press.get_value(q);
          auto p_n_times_identity        = tensor_product_u_n;
          p_n_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d)
            p_n_times_identity[d][d] = p_n;
          phi.submit_value(1.0/(gamma*dt)*u_n, q);
          phi.submit_gradient(-a21/Re*grad_u_n + 0.5*a21*tensor_product_u_n + p_n_times_identity, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi(data, 0), phi_old(data, 0), phi_int(data, 0);
      FEEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number> phi_old_press(data, 1);

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
          phi.submit_value(u_n_gamma, q);
          phi.submit_gradient((1.0 - gamma)*dt*(a32*tensor_product_u_n_gamma + a31*tensor_product_u_n -
                              a32/Re*grad_u_n_gamma - a31/Re*grad_u_n + p_n_times_identity), q);
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
                                                                     phi_old_p(data, true, 0), phi_old_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number> phi_old_press_p(data, true, 1), phi_old_press_m(data, false, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_old_p.reinit(face);
        phi_old_p.gather_evaluate(src[0], true, true);
        phi_old_m.reinit(face);
        phi_old_m.gather_evaluate(src[0], true, true);
        phi_old_press_p.reinit(face);
        phi_old_press_p.gather_evaluate(src[2], true, false);
        phi_old_press_m.reinit(face);
        phi_old_press_m.gather_evaluate(src[2], true, false);
        phi_p.reinit(face);
        phi_m.reinit(face);
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus                 = phi_p.get_normal_vector(q);
          const auto& avg_grad_u_old         = 0.5*(phi_old_p.get_gradient(q) + phi_old_m.get_gradient(q));
          const auto& avg_tensor_product_u_n = 0.5*(outer_product(phi_old_p.get_value(q), phi_old_p.get_value(q)) +
                                                    outer_product(phi_old_m.get_value(q), phi_old_m.get_value(q)));
          const auto& avg_p_old               = 0.5*(phi_old_press_p.get_value(q) + phi_old_press_m.get_value(q));
          phi_p.submit_value((a21/Re*avg_grad_u_old - 0.5*a21*avg_tensor_product_u_n)*n_plus - avg_p_old*n_plus, q);
          phi_m.submit_value(-(a21/Re*avg_grad_u_old - 0.5*a21*avg_tensor_product_u_n)*n_plus + avg_p_old*n_plus, q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi_p(data, true, 0), phi_m(data, false, 0),
                                                                     phi_old_p(data, true, 0), phi_old_m(data, false, 0),
                                                                     phi_int_p(data, true, 0), phi_int_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number> phi_old_press_p(data, true, 1), phi_old_press_m(data, false, 1);

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
          phi_p.submit_value((1.0 - gamma)*dt*((a31/Re*avg_grad_u_old + a32/Re*avg_grad_u_int -
                              a31*avg_tensor_product_u_n - a32*avg_tensor_product_u_n_gamma)*n_plus - avg_p_old*n_plus), q);
          phi_m.submit_value((1.0 - gamma)*dt*(-(a31/Re*avg_grad_u_old + a32/Re*avg_grad_u_int -
                               a31*avg_tensor_product_u_n - a32*avg_tensor_product_u_n_gamma)*n_plus + avg_p_old*n_plus), q);
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
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number> phi_old_press(data, true, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_old.reinit(face);
        phi_old.gather_evaluate(src[0], true, true);
        phi_old_extr.reinit(face);
        phi_old_extr.gather_evaluate(src[1], true, false);
        phi_old_press.reinit(face);
        phi_old_press.gather_evaluate(src[2], true, false);
        phi.reinit(face);
        const auto coef_jump = C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus             = phi.get_normal_vector(q);
          const auto& grad_u_old         = phi_old.get_gradient(q);
          const auto& tensor_product_u_n = outer_product(phi_old.get_value(q), phi_old.get_value(q));
          const auto& p_old              = phi_old_press.get_value(q);
          const auto& point_vectorized   = phi.quadrature_point(q);
          auto u_int_m                   = Tensor<1, dim, VectorizedArray<Number>>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d)
              point[d] = point_vectorized[d][v];
            for(unsigned int d = 0; d < dim; ++d)
              u_int_m[d][v] = vel_exact.value(point, d);
          }
          const auto tensor_product_u_int_m   = outer_product(u_int_m, phi_old_extr.get_value(q));
          const auto lambda                   = std::abs(scalar_product(phi_old_extr.get_value(q), n_plus));
          const auto tensor_product_u_int_m_1 = outer_product(phi_old.get_value(q), u_int_m);
          const auto lambda_1                 = std::abs(scalar_product(phi_old.get_value(q), n_plus));
          const auto tensor_product_u_int_m_2 = outer_product(u_int_m, phi_old.get_value(q));
          phi.submit_value((a21/Re*grad_u_old - 0.5*a21*tensor_product_u_n)*n_plus - p_old*n_plus +
                           a22/Re*2.0*coef_jump*u_int_m -
                           0.5*a22*tensor_product_u_int_m*n_plus + 0.5*a22*lambda*u_int_m -
                           0.5*a22*tensor_product_u_int_m_1*n_plus + 0.5*a22*lambda_1*u_int_m -
                           0.5*a22*tensor_product_u_int_m_2*n_plus + 0.5*a22*lambda_1*u_int_m , q);
          phi.submit_normal_derivative(-theta_v*a22/Re*u_int_m, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi(data, true, 0),
                                                                     phi_old(data, true, 0),
                                                                     phi_int(data, true, 0),
                                                                     phi_int_extr(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi_old_press(data, true, 1);

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
        const auto coef_jump = C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus                   = phi.get_normal_vector(q);
          const auto& grad_u_old               = phi_old.get_gradient(q);
          const auto& grad_u_int               = phi_int.get_gradient(q);
          const auto& tensor_product_u_n       = outer_product(phi_old.get_value(q), phi_old.get_value(q));
          const auto& tensor_product_u_n_gamma = outer_product(phi_int.get_value(q), phi_int.get_value(q));
          const auto& p_old                    = phi_old_press.get_value(q);
          const auto& point_vectorized         = phi.quadrature_point(q);
          auto u_m                             = Tensor<1, dim, VectorizedArray<Number>>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d)
              point[d] = point_vectorized[d][v];
            for(unsigned int d = 0; d < dim; ++d)
              u_m[d][v] = vel_exact.value(point, d);
          }
          const auto tensor_product_u_m = outer_product(u_m, phi_int_extr.get_value(q));
          const auto lambda             = std::abs(scalar_product(phi_int_extr.get_value(q), n_plus));
          phi.submit_value((1.0 - gamma)*dt*((a31/Re*grad_u_old + a32/Re*grad_u_int -
                           a31*tensor_product_u_n - a32*tensor_product_u_n_gamma)*n_plus - p_old*n_plus +
                           a33/Re*2.0*coef_jump*u_m -
                           a33*tensor_product_u_m*n_plus + a33*lambda*u_m), q);
          phi.submit_normal_derivative((1.0 - gamma)*dt*(-theta_v*a33/Re*u_m), q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
  }


  // Put together all the previous steps for velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
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
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_rhs_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_p, n_q_points_1d - 1, 1, Number>   phi(data, 1, 1), phi_old(data, 1, 1);
    FEEvaluation<dim, fe_degree_v, n_q_points_1d - 1, dim, Number> phi_proj(data, 0, 1);

    const double coeff_2 = (TR_BDF2_stage == 1) ? 1e6*gamma*dt : 1e6*(1.0 - gamma)*dt;

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_proj.reinit(cell);
      phi_proj.gather_evaluate(src[0], true, false);
      phi_old.reinit(cell);
      phi_old.gather_evaluate(src[1], true, false);
      phi.reinit(cell);
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& u_star_star = phi_proj.get_value(q);
        const auto& p_old       = phi_old.get_value(q);
        phi.submit_value(p_old, q);
        phi.submit_gradient(coeff_2*u_star_star, q);
      }
      phi.integrate_scatter(true, true, dst);
    }
  }


  // Assemble rhs face term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_rhs_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d - 1, 1, Number>   phi_p(data, true, 1, 1), phi_m(data, false, 1, 1);
    FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d - 1, dim, Number> phi_proj_p(data, true, 0, 1), phi_proj_m(data, false, 0, 1);

    const double coeff = (TR_BDF2_stage == 1) ? 1e6*gamma*dt : 1e6*(1.0 - gamma)*dt;

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
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_rhs_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                      Vec&                                         dst,
                                      const std::vector<Vec>&                      src,
                                      const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d - 1, 1, Number>   phi(data, true, 1, 1);
    FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d - 1, dim, Number> phi_proj(data, true, 0, 1);

    const double coeff = (TR_BDF2_stage == 1) ? 1e6*gamma*dt : 1e6*(1.0 - gamma)*dt;

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
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
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
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(TR_BDF2_stage == 1) {
      FEEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi(data, 0), phi_old(data, 0), phi_old_extr(data, 0);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi.reinit(cell);
        phi.gather_evaluate(src, true, true);
        phi_old.reinit(cell);
        phi_old.gather_evaluate(u_n, true, false);
        phi_old_extr.reinit(cell);
        phi_old_extr.gather_evaluate(u_extr, true, false);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& u_int                  = phi.get_value(q);
          const auto& grad_u_int             = phi.get_gradient(q);
          const auto& u_old                  = phi_old.get_value(q);
          const auto& u_n_gamma_ov_2         = phi_old_extr.get_value(q);
          const auto& tensor_product_u_int   = outer_product(u_int, u_n_gamma_ov_2);
          const auto& tensor_product_u_int_1 = outer_product(u_int, u_old);
          const auto& tensor_product_u_int_2 = outer_product(u_old, u_int);
          phi.submit_value(1.0/(gamma*dt)*u_int, q);
          phi.submit_gradient(-0.5*a22*tensor_product_u_int   +
                              -0.5*a22*tensor_product_u_int_1 +
                              -0.5*a22*tensor_product_u_int_2 +
                              a22/Re*grad_u_int, q);
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
          const auto& u_curr                = phi.get_value(q);
          const auto& grad_u_curr           = phi.get_gradient(q);
          const auto& u_n1_int              = phi_int_extr.get_value(q);
          const auto& tensor_product_u_curr = outer_product(u_curr, u_n1_int);
          phi.submit_value(u_curr, q);
          phi.submit_gradient((1.0 - gamma)*dt*(-a33*tensor_product_u_curr + a33/Re*grad_u_curr), q);
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
                                                                     phi_old_p(data, true, 0), phi_old_m(data, false, 0),
                                                                     phi_old_extr_p(data, true, 0), phi_old_extr_m(data, false, 0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_p.reinit(face);
        phi_p.gather_evaluate(src, true, true);
        phi_m.reinit(face);
        phi_m.gather_evaluate(src, true, true);
        const auto coef_jump = C_u*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                        std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));
        phi_old_p.reinit(face);
        phi_old_p.gather_evaluate(u_n, true, false);
        phi_old_m.reinit(face);
        phi_old_m.gather_evaluate(u_n, true, false);
        phi_old_extr_p.reinit(face);
        phi_old_extr_p.gather_evaluate(u_extr, true, false);
        phi_old_extr_m.reinit(face);
        phi_old_extr_m.gather_evaluate(u_extr, true, false);
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus                     = phi_p.get_normal_vector(q);
          const auto& avg_grad_u_int             = 0.5*(phi_p.get_gradient(q) + phi_m.get_gradient(q));
          const auto& jump_u_int                 = phi_p.get_value(q) - phi_m.get_value(q);
          const auto& avg_tensor_product_u_int   = 0.5*(outer_product(phi_p.get_value(q), phi_old_extr_p.get_value(q)) +
                                                        outer_product(phi_m.get_value(q), phi_old_extr_m.get_value(q)));
          const auto  lambda                     = std::max(std::abs(scalar_product(phi_old_extr_p.get_value(q), n_plus)),
                                                            std::abs(scalar_product(phi_old_extr_m.get_value(q), n_plus)));
          const auto& avg_tensor_product_u_int_1 = 0.5*(outer_product(phi_p.get_value(q), phi_old_p.get_value(q)) +
                                                        outer_product(phi_m.get_value(q), phi_old_m.get_value(q)));
          const auto  lambda_1                   = std::max(std::abs(scalar_product(phi_old_p.get_value(q), n_plus)),
                                                            std::abs(scalar_product(phi_old_m.get_value(q), n_plus)));
          const auto& avg_tensor_product_u_int_2 = 0.5*(outer_product(phi_old_p.get_value(q), phi_p.get_value(q)) +
                                                        outer_product(phi_old_m.get_value(q), phi_m.get_value(q)));
          phi_p.submit_value(a22/Re*(-avg_grad_u_int*n_plus + coef_jump*jump_u_int) +
                             0.5*a22*avg_tensor_product_u_int*n_plus + 0.5*0.5*a22*lambda*jump_u_int +
                             0.5*a22*avg_tensor_product_u_int_1*n_plus + 0.5*0.5*a22*lambda_1*jump_u_int +
                             0.5*a22*avg_tensor_product_u_int_2*n_plus + 0.5*0.5*a22*lambda_1*jump_u_int, q);
          phi_m.submit_value(-a22/Re*(-avg_grad_u_int*n_plus + coef_jump*jump_u_int) -
                              0.5*a22*avg_tensor_product_u_int*n_plus - 0.5*0.5*a22*lambda*jump_u_int -
                              0.5*a22*avg_tensor_product_u_int_1*n_plus - 0.5*0.5*a22*lambda_1*jump_u_int -
                              0.5*a22*avg_tensor_product_u_int_2*n_plus - 0.5*0.5*a22*lambda_1*jump_u_int, q);
          phi_p.submit_normal_derivative(-theta_v*a22/Re*0.5*jump_u_int, q);
          phi_m.submit_normal_derivative(-theta_v*a22/Re*0.5*jump_u_int, q);
        }
        phi_p.integrate_scatter(true, true, dst);
        phi_m.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi_p(data, true, 0), phi_m(data, false, 0),
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
          phi_p.submit_value((1.0 - gamma)*dt*(a33/Re*(-avg_grad_u*n_plus + coef_jump*jump_u) +
                             a33*avg_tensor_product_u*n_plus + 0.5*a33*lambda*jump_u), q);
          phi_m.submit_value((1.0 - gamma)*dt*(-a33/Re*(-avg_grad_u*n_plus + coef_jump*jump_u) -
                              a33*avg_tensor_product_u*n_plus - 0.5*a33*lambda*jump_u), q);
          phi_p.submit_normal_derivative((1.0 - gamma)*dt*(-theta_v*a33/Re*0.5*jump_u), q);
          phi_m.submit_normal_derivative((1.0 - gamma)*dt*(-theta_v*a33/Re*0.5*jump_u), q);
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
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi(data, true, 0), phi_old(data, true, 0),
                                                                     phi_old_extr(data, true, 0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi.reinit(face);
        phi.gather_evaluate(src, true, true);
        const auto coef_jump  = C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);
        const double coef_trasp = 0.0;
        phi_old.reinit(face);
        phi_old.gather_evaluate(u_n, true, false);
        phi_old_extr.reinit(face);
        phi_old_extr.gather_evaluate(u_extr, true, false);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus                 = phi.get_normal_vector(q);
          const auto& tensor_product_n       = outer_product(n_plus, n_plus);
          const auto& grad_u_int             = phi.get_gradient(q);
          const auto& u_int                  = phi.get_value(q);
          const auto& tensor_product_u_int   = outer_product(phi.get_value(q), phi_old_extr.get_value(q));
          const auto  lambda                 = std::abs(scalar_product(phi_old_extr.get_value(q), n_plus));
          const auto& tensor_product_u_int_1 = outer_product(phi.get_value(q), phi_old.get_value(q));
          const auto  lambda_1               = std::abs(scalar_product(phi_old.get_value(q), n_plus));
          const auto& tensor_product_u_int_2 = outer_product(phi_old.get_value(q), phi.get_value(q));
          phi.submit_value(a22/Re*(-grad_u_int*n_plus + 2.0*coef_jump*u_int) +
                           0.5*a22*coef_trasp*tensor_product_u_int*n_plus + 0.5*a22*lambda*u_int +
                           0.5*a22*coef_trasp*tensor_product_u_int_1*n_plus + 0.5*a22*lambda_1*u_int +
                           0.5*a22*coef_trasp*tensor_product_u_int_2*n_plus + 0.5*a22*lambda_1*u_int, q);
          phi.submit_normal_derivative(-theta_v*a22/Re*u_int, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi(data, true, 0), phi_extr(data, true, 0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi.reinit(face);
        phi.gather_evaluate(src, true, true);
        const auto coef_jump  = C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);
        const double coef_trasp = 0.0;
        phi_extr.reinit(face);
        phi_extr.gather_evaluate(u_extr, true, false);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus           = phi.get_normal_vector(q);
          const auto& tensor_product_n = outer_product(n_plus, n_plus);
          const auto& grad_u           = phi.get_gradient(q);
          const auto& u                = phi.get_value(q);
          const auto& tensor_product_u = outer_product(phi.get_value(q), phi_extr.get_value(q));
          const auto  lambda           = std::abs(scalar_product(phi_extr.get_value(q), n_plus));
          phi.submit_value((1.0 - gamma)*dt*(a33/Re*(-grad_u*n_plus + 2.0*coef_jump*u) +
                           a33*coef_trasp*tensor_product_u*n_plus + a33*lambda*u), q);
          phi.submit_normal_derivative((1.0 - gamma)*dt*(-theta_v*a33/Re*u), q);
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
    FEEvaluation<dim, fe_degree_p, n_q_points_1d - 1, 1, Number> phi(data, 1, 1);

    const double coeff = (TR_BDF2_stage == 1) ? 1e6*gamma*dt*gamma*dt : 1e6*(1.0 - gamma)*dt*(1.0 - gamma)*dt;

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, true, true);
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        phi.submit_gradient(coeff*phi.get_gradient(q), q);
        phi.submit_value(phi.get_value(q), q);
      }
      phi.integrate_scatter(true, true, dst);
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
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d - 1, 1, Number> phi_p(data, true, 1, 1), phi_m(data, false, 1, 1);

    const double coeff = (TR_BDF2_stage == 1) ? 1e6*gamma*dt*gamma*dt : 1e6*(1.0 - gamma)*dt*(1.0 - gamma)*dt;

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
        phi_p.submit_value(coeff*(-scalar_product(avg_grad_pres, n_plus) + coef_jump*jump_pres), q);
        phi_m.submit_value(coeff*(scalar_product(avg_grad_pres, n_plus) - coef_jump*jump_pres), q);
        phi_p.submit_gradient(coeff*(-theta_p*0.5*jump_pres*n_plus), q);
        phi_m.submit_gradient(coeff*(-theta_p*0.5*jump_pres*n_plus), q);
      }
      phi_p.integrate_scatter(true, true, dst);
      phi_m.integrate_scatter(true, true, dst);
    }
  }


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
    else if(NS_stage == 3) {
      this->data->cell_loop(&NavierStokesProjectionOperator::assemble_cell_term_projection_grad_p,
                            this, dst, src, false);
    }
    else
      Assert(false, ExcNotImplemented());
  }


  // Assemble rhs cell term for the projection of gradient of pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_cell_term_projection_grad_p(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const Vec&                                   src,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi(data, 0);

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
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_rhs_cell_term_projection_grad_p(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const Vec&                                   src,
                                           const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi(data, 0);
    FEEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi_pres(data, 1);

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
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  vmult_grad_p_projection(Vec& dst, const Vec& src) const {
    this->data->cell_loop(&NavierStokesProjectionOperator::assemble_rhs_cell_term_projection_grad_p,
                          this, dst, src, true);
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
      FEEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi(data, 0), phi_old(data, 0), phi_old_extr(data, 0);

      AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);
      Tensor<1, dim, VectorizedArray<Number>> tmp;
      for(unsigned int d = 0; d < dim; ++d)
        tmp[d] = make_vectorized_array<Number>(1.0);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_old.reinit(cell);
        phi_old.gather_evaluate(u_n, true, false);
        phi_old_extr.reinit(cell);
        phi_old_extr.gather_evaluate(u_extr, true, false);
        phi.reinit(cell);
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
            phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
          phi.submit_dof_value(tmp, i);
          phi.evaluate(true, true);
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& u_int                  = phi.get_value(q);
            const auto& grad_u_int             = phi.get_gradient(q);
            const auto& u_old                  = phi_old.get_value(q);
            const auto& u_n_gamma_ov_2         = phi_old_extr.get_value(q);
            const auto& tensor_product_u_int   = outer_product(u_int, u_n_gamma_ov_2);
            const auto& tensor_product_u_int_1 = outer_product(u_int, u_old);
            const auto& tensor_product_u_int_2 = outer_product(u_old, u_int);
            phi.submit_value(1.0/(gamma*dt)*u_int, q);
            phi.submit_gradient(-0.5*a22*tensor_product_u_int   +
                                -0.5*a22*tensor_product_u_int_1 +
                                -0.5*a22*tensor_product_u_int_2 +
                                a22/Re*grad_u_int, q);
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
            phi.submit_value(u_curr, q);
            phi.submit_gradient((1.0 - gamma)*dt*(-a33*tensor_product_u_curr + a33/Re*grad_u_curr), q);
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
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_diagonal_face_term_velocity(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const Vec&                                   src,
                                       const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi_p(data, true, 0), phi_m(data, false, 0),
                                                                     phi_old_p(data, true, 0), phi_old_m(data, false, 0),
                                                                     phi_old_extr_p(data, true, 0), phi_old_extr_m(data, false, 0);

      AssertDimension(phi_p.dofs_per_component, phi_m.dofs_per_component);
      AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal_p(phi_p.dofs_per_component), diagonal_m(phi_m.dofs_per_component);
      Tensor<1, dim, VectorizedArray<Number>> tmp;
      for(unsigned int d = 0; d < dim; ++d)
        tmp[d] = make_vectorized_array<Number>(1.0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_old_p.reinit(face);
        phi_old_p.gather_evaluate(u_n, true, false);
        phi_old_m.reinit(face);
        phi_old_m.gather_evaluate(u_n, true, false);
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
            const auto& n_plus                     = phi_p.get_normal_vector(q);
            const auto& avg_grad_u_int             = 0.5*(phi_p.get_gradient(q) + phi_m.get_gradient(q));
            const auto& jump_u_int                 = phi_p.get_value(q) - phi_m.get_value(q);
            const auto& avg_tensor_product_u_int   = 0.5*(outer_product(phi_p.get_value(q), phi_old_extr_p.get_value(q)) +
                                                        outer_product(phi_m.get_value(q), phi_old_extr_m.get_value(q)));
            const auto  lambda                     = std::max(std::abs(scalar_product(phi_old_extr_p.get_value(q), n_plus)),
                                                            std::abs(scalar_product(phi_old_extr_m.get_value(q), n_plus)));
            const auto& avg_tensor_product_u_int_1 = 0.5*(outer_product(phi_p.get_value(q), phi_old_p.get_value(q)) +
                                                          outer_product(phi_m.get_value(q), phi_old_m.get_value(q)));
            const auto  lambda_1                   = std::max(std::abs(scalar_product(phi_old_p.get_value(q), n_plus)),
                                                              std::abs(scalar_product(phi_old_m.get_value(q), n_plus)));
            const auto& avg_tensor_product_u_int_2 = 0.5*(outer_product(phi_old_p.get_value(q), phi_p.get_value(q)) +
                                                          outer_product(phi_old_m.get_value(q), phi_m.get_value(q)));
            phi_p.submit_value(a22/Re*(-avg_grad_u_int*n_plus + coef_jump*jump_u_int) +
                               0.5*a22*avg_tensor_product_u_int*n_plus + 0.5*0.5*a22*lambda*jump_u_int +
                               0.5*a22*avg_tensor_product_u_int_1*n_plus + 0.5*0.5*a22*lambda_1*jump_u_int +
                               0.5*a22*avg_tensor_product_u_int_2*n_plus + 0.5*0.5*a22*lambda_1*jump_u_int, q);
            phi_m.submit_value(-a22/Re*(-avg_grad_u_int*n_plus + coef_jump*jump_u_int) -
                                0.5*a22*avg_tensor_product_u_int*n_plus - 0.5*0.5*a22*lambda*jump_u_int -
                                0.5*a22*avg_tensor_product_u_int_1*n_plus - 0.5*0.5*a22*lambda_1*jump_u_int -
                                0.5*a22*avg_tensor_product_u_int_2*n_plus - 0.5*0.5*a22*lambda_1*jump_u_int, q);
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
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi_p(data, true, 0), phi_m(data, false, 0),
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
            phi_p.submit_value((1.0 - gamma)*dt*(a33/Re*(-avg_grad_u*n_plus + coef_jump*jump_u) +
                               a33*avg_tensor_product_u*n_plus + 0.5*a33*lambda*jump_u), q);
            phi_m.submit_value((1.0 - gamma)*dt*(-a33/Re*(-avg_grad_u*n_plus + coef_jump*jump_u) -
                               a33*avg_tensor_product_u*n_plus - 0.5*a33*lambda*jump_u), q);
            phi_p.submit_normal_derivative((1.0 - gamma)*dt*(-theta_v*0.5*a33/Re*jump_u), q);
            phi_m.submit_normal_derivative((1.0 - gamma)*dt*(-theta_v*0.5*a33/Re*jump_u), q);
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
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_diagonal_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const Vec&                                   src,
                                           const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi(data, true, 0), phi_old(data, true, 0),
                                                                     phi_old_extr(data, true, 0);

      AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);
      Tensor<1, dim, VectorizedArray<Number>> tmp;
      for(unsigned int d = 0; d < dim; ++d)
        tmp[d] = make_vectorized_array<Number>(1.0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_old.reinit(face);
        phi_old.gather_evaluate(u_n, true, false);
        phi_old_extr.reinit(face);
        phi_old_extr.gather_evaluate(u_extr, true, false);
        phi.reinit(face);
        const auto coef_jump    = C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);
        const double coef_trasp = 0.0;
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
            phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
          phi.submit_dof_value(tmp, i);
          phi.evaluate(true, true);
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& n_plus                 = phi.get_normal_vector(q);
            const auto& tensor_product_n       = outer_product(n_plus, n_plus);
            const auto& grad_u_int             = phi.get_gradient(q);
            const auto& u_int                  = phi.get_value(q);
            const auto& tensor_product_u_int   = outer_product(phi.get_value(q), phi_old_extr.get_value(q));
            const auto  lambda                 = std::abs(scalar_product(phi_old_extr.get_value(q), n_plus));
            const auto& tensor_product_u_int_1 = outer_product(phi.get_value(q), phi_old.get_value(q));
            const auto  lambda_1               = std::abs(scalar_product(phi_old.get_value(q), n_plus));
            const auto& tensor_product_u_int_2 = outer_product(phi_old.get_value(q), phi.get_value(q));
            phi.submit_value(a22/Re*(-grad_u_int*n_plus + 2.0*coef_jump*u_int) +
                             0.5*a22*coef_trasp*tensor_product_u_int*n_plus + 0.5*a22*lambda*u_int +
                             0.5*a22*coef_trasp*tensor_product_u_int_1*n_plus + 0.5*a22*lambda_1*u_int +
                             0.5*a22*coef_trasp*tensor_product_u_int_2*n_plus + 0.5*a22*lambda_1*u_int, q);
            phi.submit_normal_derivative(-theta_v*a22/Re*u_int, q);
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
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d, dim, Number> phi(data, true, 0), phi_extr(data, true, 0);

      AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);
      Tensor<1, dim, VectorizedArray<Number>> tmp;
      for(unsigned int d = 0; d < dim; ++d)
        tmp[d] = make_vectorized_array<Number>(1.0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_extr.reinit(face);
        phi_extr.gather_evaluate(u_extr, true, false);
        phi.reinit(face);
        const auto coef_jump    = C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);
        const double coef_trasp = 0.0;
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
            phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
          phi.submit_dof_value(tmp, i);
          phi.evaluate(true, true);
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& n_plus           = phi.get_normal_vector(q);
            const auto& tensor_product_n = outer_product(n_plus, n_plus);
            const auto& grad_u           = phi.get_gradient(q);
            const auto& u                = phi.get_value(q);
            const auto& tensor_product_u = outer_product(phi.get_value(q), phi_extr.get_value(q));
            const auto  lambda           = std::abs(scalar_product(phi_extr.get_value(q), n_plus));
            phi.submit_value((1.0 - gamma)*dt*(a33/Re*(-grad_u*n_plus + coef_jump*u) +
                             a33*coef_trasp*tensor_product_u*n_plus + a33*lambda*u), q);
            phi.submit_normal_derivative((1.0 - gamma)*dt*(-theta_v*a33/Re*u), q);
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
    FEEvaluation<dim, fe_degree_p, n_q_points_1d - 1, 1, Number> phi(data, 1, 1);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    const double coeff = (TR_BDF2_stage == 1) ? 1e6*gamma*dt*gamma*dt : 1e6*(1.0 - gamma)*dt*(1.0 - gamma)*dt;

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
          phi.submit_dof_value(VectorizedArray<Number>(), j);
        phi.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        phi.evaluate(true, true);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          phi.submit_value(phi.get_value(q), q);
          phi.submit_gradient(coeff*phi.get_gradient(q), q);
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
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::
  assemble_diagonal_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const Vec&                                   src,
                                       const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d - 1, 1, Number> phi_p(data, true, 1, 1), phi_m(data, false, 1, 1);

    AssertDimension(phi_p.dofs_per_component, phi_m.dofs_per_component);
    AlignedVector<VectorizedArray<Number>> diagonal_p(phi_p.dofs_per_component),
                                           diagonal_m(phi_m.dofs_per_component);

    const double coeff = (TR_BDF2_stage == 1) ? 1e6*gamma*dt*gamma*dt : 1e6*(1.0 - gamma)*dt*(1.0 - gamma)*dt;

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
          phi_p.submit_value(coeff*(-scalar_product(avg_grad_pres, n_plus) + coef_jump*jump_pres), q);
          phi_m.submit_value(coeff*(scalar_product(avg_grad_pres, n_plus) - coef_jump*jump_pres), q);
          phi_p.submit_gradient(coeff*(-theta_p*0.5*jump_pres*n_plus), q);
          phi_m.submit_gradient(coeff*(-theta_p*0.5*jump_pres*n_plus), q);
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
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d, typename Vec, typename Number>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d, Vec, Number>::compute_diagonal() {
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
    const double       CFL;

    EquationData::Velocity<dim> vel_exact;
    EquationData::Pressure<dim> pres_exact;

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

    DeclException2(ExcInvalidTimeStep,
                   double,
                   double,
                   << " The time step " << arg1 << " is out of range."
                   << std::endl
                   << " The permitted range is (0," << arg2 << "]");

    void create_triangulation_and_dofs(const unsigned int n_refines);

    void initialize();

    void diffusion_step();

    void projection_step();

    void project_grad(const unsigned int flag);

    void analyze_results();

    void output_results(const unsigned int step);

    void output_errors(const unsigned int step);

  private:
    std::shared_ptr<MatrixFree<dim, double>> matrix_free_storage;

    NavierStokesProjectionOperator<dim, EquationData::degree_p, EquationData::degree_p + 1,
                                   EquationData::degree_p + 2,
                                   LinearAlgebra::distributed::Vector<double>, double> navier_stokes_matrix;

    AffineConstraints<double> constraints_velocity, constraints_pressure;

    unsigned int vel_max_its;
    unsigned int vel_Krylov_size;
    unsigned int vel_off_diagonals;
    unsigned int vel_update_prec;
    double       vel_eps;
    double       vel_diag_strength;

    Vector<double> H1_error_per_cell_vel, L2_error_per_cell_vel, Linfty_error_per_cell_vel,
                   H1_rel_error_per_cell_vel, L2_rel_error_per_cell_vel, Linfty_rel_error_per_cell_vel,
                   L2_error_per_cell_pres, Linfty_error_per_cell_pres,
                   L2_rel_error_per_cell_pres, Linfty_rel_error_per_cell_pres;

    double dt;

    std::string saving_dir;

    std::ofstream output_error_pres;
    std::ofstream output_error_vel;

    ConditionalOStream pcout;

    std::ofstream      time_out;
    ConditionalOStream ptime_out;
    TimerOutput        time_table;
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
    gamma(1.0),  //--- Save also in the NavierStokes class the TR-BDF2 parameter value
    TR_BDF2_stage(1),             //--- Initialize the flag for the TR_BDF2 stage
    Re(data.Reynolds),
    CFL(data.CFL),
    vel_exact(data.Reynolds, data.initial_time),
    pres_exact(data.Reynolds, data.initial_time),
    triangulation(MPI_COMM_WORLD),
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
    dt(5e-4),
    saving_dir(data.dir),
    output_error_pres("./" + data.dir + "/error_analysis_pres.dat", std::ofstream::out),
    output_error_vel("./" + data.dir + "/error_analysis_vel.dat", std::ofstream::out),
    pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    time_out("./" + data.dir + "/time_analysis_" +
             Utilities::int_to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)) + "proc.dat"),
    ptime_out(time_out, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    time_table(ptime_out, TimerOutput::summary, TimerOutput::cpu_and_wall_times) {
      if(EquationData::degree_p < 1) {
        pcout
        << " WARNING: The chosen pair of finite element spaces is not stable."
        << std::endl
        << " The obtained results will be nonsense" << std::endl;
      }

      AssertThrow(!((dt <= 0.0) || (dt > 0.5*T)), ExcInvalidTimeStep(dt, 0.5*T));

      constraints_velocity.clear();
      constraints_pressure.clear();
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
    TimerOutput::Scope t(time_table, "Create triangulation and dofs");

    GridGenerator::hyper_cube(triangulation, 0.0, 2.0*numbers::PI, true);

    pcout << "Number of refines = " << n_refines << std::endl;
    triangulation.refine_global(n_refines);
    pcout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
    dt = CFL*2.0*numbers::PI/(std::pow(2, n_refines))/(EquationData::degree_p + 1);
    navier_stokes_matrix.set_dt(dt);

    Vector<double> error_per_cell_tmp(triangulation.n_active_cells());
    H1_error_per_cell_vel.reinit(error_per_cell_tmp);
    L2_error_per_cell_vel.reinit(error_per_cell_tmp);
    Linfty_error_per_cell_vel.reinit(error_per_cell_tmp);
    H1_rel_error_per_cell_vel.reinit(error_per_cell_tmp);
    L2_rel_error_per_cell_vel.reinit(error_per_cell_tmp);
    Linfty_rel_error_per_cell_vel.reinit(error_per_cell_tmp);

    L2_error_per_cell_pres.reinit(error_per_cell_tmp);
    Linfty_error_per_cell_pres.reinit(error_per_cell_tmp);
    L2_rel_error_per_cell_pres.reinit(error_per_cell_tmp);
    Linfty_rel_error_per_cell_pres.reinit(error_per_cell_tmp);

    dof_handler_velocity.distribute_dofs(fe_velocity);
    dof_handler_pressure.distribute_dofs(fe_pressure);

    pcout << "dim (X_h) = " << dof_handler_velocity.n_dofs()
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

    std::vector<QGauss<1>> quadratures;
    quadratures.push_back(QGauss<1>(EquationData::degree_p + 2));
    quadratures.push_back(QGauss<1>(EquationData::degree_p + 1));

    matrix_free_storage = std::make_unique<MatrixFree<dim, double>>();
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
  }


  // @sect4{ <code>NavierStokesProjection::initialize</code> }

  // This method loads the initial data
  //
  template<int dim>
  void NavierStokesProjection<dim>::initialize() {
    TimerOutput::Scope t(time_table, "Initialize pressure and velocity");

    pres_exact.set_time(t_0 + dt);
    VectorTools::interpolate(dof_handler_pressure, pres_exact, pres_n);

    vel_exact.set_time(t_0);
    VectorTools::interpolate(dof_handler_velocity, vel_exact, u_n_minus_1);
    vel_exact.advance_time(dt);
    VectorTools::interpolate(dof_handler_velocity, vel_exact, u_n);
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

    navier_stokes_matrix.set_time(vel_exact.get_time());

    navier_stokes_matrix.set_NS_stage(1);

    if(TR_BDF2_stage == 1) {
      navier_stokes_matrix.vmult_rhs_velocity(rhs_u, {u_n, u_n, pres_n});
      navier_stokes_matrix.set_u_n(u_n);
      u_n_k = u_n;
    }
    else {
      navier_stokes_matrix.vmult_rhs_velocity(rhs_u, {u_n, u_n_gamma, pres_int, u_n_gamma});
      u_n_gamma_k = u_extr;
    }

    for(unsigned int iter = 0; iter < 100; ++iter) {
      if(TR_BDF2_stage == 1) {
        navier_stokes_matrix.set_u_extr(u_n_k);
        u_star = u_n_k;
      }
      else {
        navier_stokes_matrix.set_u_extr(u_n_gamma_k);
        u_star = u_n_gamma_k;
      }

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
    PreconditionJacobi<NavierStokesProjectionOperator<dim,
                                                      EquationData::degree_p,
                                                      EquationData::degree_p + 1,
                                                      EquationData::degree_p + 2,
                                                      LinearAlgebra::distributed::Vector<double>,
                                                      double>> preconditioner;
    navier_stokes_matrix.compute_diagonal();
    preconditioner.initialize(navier_stokes_matrix);
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


  // Since we have solved a problem with analytic solution, we want to verify
  // the correctness of our implementation by computing the errors of the
  // numerical result against the analytic solution.
  //
  template <int dim>
  void NavierStokesProjection<dim>::analyze_results() {
    TimerOutput::Scope t(time_table, "Analysis results: computing errrors");

    u_tmp = 0;
    pres_int = 0;

    VectorTools::integrate_difference(dof_handler_velocity, u_n, vel_exact,
                                      H1_error_per_cell_vel, quadrature_velocity, VectorTools::H1_norm);
    const double error_vel_H1 = VectorTools::compute_global_error(triangulation, H1_error_per_cell_vel, VectorTools::H1_norm);
    VectorTools::integrate_difference(dof_handler_velocity, u_tmp, vel_exact,
                                      H1_rel_error_per_cell_vel, quadrature_velocity, VectorTools::H1_norm);
    const double H1_vel = VectorTools::compute_global_error(triangulation, H1_rel_error_per_cell_vel, VectorTools::H1_norm);
    const double error_rel_vel_H1 = error_vel_H1/H1_vel;
    pcout << "Verification via H1 error velocity:    "<< error_vel_H1 << std::endl;
    pcout << "Verification via H1 relative error velocity:    "<< error_rel_vel_H1 << std::endl;

    VectorTools::integrate_difference(dof_handler_velocity, u_n, vel_exact,
                                      L2_error_per_cell_vel, quadrature_velocity, VectorTools::L2_norm);
    const double error_vel_L2 = VectorTools::compute_global_error(triangulation, L2_error_per_cell_vel, VectorTools::L2_norm);
    VectorTools::integrate_difference(dof_handler_velocity, u_tmp, vel_exact,
                                      L2_rel_error_per_cell_vel, quadrature_velocity, VectorTools::L2_norm);
    const double L2_vel = VectorTools::compute_global_error(triangulation, L2_rel_error_per_cell_vel, VectorTools::L2_norm);
    const double error_rel_vel_L2 = error_vel_L2/L2_vel;
    pcout << "Verification via L2 error velocity:    "<< error_vel_L2 << std::endl;
    pcout << "Verification via L2 relative error velocity:    "<< error_rel_vel_L2 << std::endl;

    VectorTools::integrate_difference(dof_handler_velocity, u_n, vel_exact,
                                      Linfty_error_per_cell_vel, quadrature_velocity, VectorTools::Linfty_norm);
    const double error_vel_Linfty = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_vel, VectorTools::Linfty_norm);
    VectorTools::integrate_difference(dof_handler_velocity, u_tmp, vel_exact,
                                      Linfty_rel_error_per_cell_vel, quadrature_velocity, VectorTools::Linfty_norm);
    const double Linfty_vel = VectorTools::compute_global_error(triangulation, Linfty_rel_error_per_cell_vel, VectorTools::Linfty_norm);
    const double error_rel_vel_Linfty = error_vel_Linfty/Linfty_vel;
    pcout << "Verification via Linfty error velocity:    "<< error_vel_Linfty << std::endl;
    pcout << "Verification via Linfty relative error velocity:    "<< error_rel_vel_Linfty << std::endl;

    VectorTools::integrate_difference(dof_handler_pressure, pres_n, pres_exact,
                                      L2_error_per_cell_pres, quadrature_pressure, VectorTools::L2_norm);
    const double error_pres_L2 = VectorTools::compute_global_error(triangulation, L2_error_per_cell_pres, VectorTools::L2_norm);
    VectorTools::integrate_difference(dof_handler_pressure, pres_int, pres_exact,
                                      L2_rel_error_per_cell_pres, quadrature_pressure, VectorTools::L2_norm);
    const double pres_L2 = VectorTools::compute_global_error(triangulation, L2_rel_error_per_cell_pres, VectorTools::L2_norm);
    const double error_rel_pres_L2 = error_pres_L2/pres_L2;
    pcout << "Verification via L2 error pressure:    "<< error_pres_L2 << std::endl;
    pcout << "Verification via L2 relative error pressure:    "<< error_rel_pres_L2 << std::endl;

    VectorTools::integrate_difference(dof_handler_pressure, pres_n, pres_exact,
                                      Linfty_error_per_cell_pres, quadrature_pressure, VectorTools::Linfty_norm);
    const double error_pres_Linfty =
    VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_pres, VectorTools::Linfty_norm);
    VectorTools::integrate_difference(dof_handler_pressure, pres_int, pres_exact,
                                      Linfty_rel_error_per_cell_pres, quadrature_pressure, VectorTools::Linfty_norm);
    const double pres_Linfty = VectorTools::compute_global_error(triangulation, Linfty_rel_error_per_cell_pres, VectorTools::Linfty_norm);
    const double error_rel_pres_Linfty = error_pres_Linfty/pres_Linfty;
    pcout << "Verification via Linfty error pressure:    "<< error_pres_Linfty << std::endl;
    pcout << "Verification via Linfty relative error pressure:    "<< error_rel_pres_Linfty << std::endl;

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
      output_error_vel << error_vel_H1 << std::endl;
      output_error_vel << error_rel_vel_H1 << std::endl;
      output_error_vel << error_vel_L2 << std::endl;
      output_error_vel << error_rel_vel_L2 << std::endl;
      output_error_vel << error_vel_Linfty << std::endl;
      output_error_vel << error_rel_vel_Linfty << std::endl;

      output_error_pres << error_pres_L2 << std::endl;
      output_error_pres << error_rel_pres_L2 << std::endl;
      output_error_pres << error_pres_Linfty << std::endl;
      output_error_pres << error_rel_pres_Linfty << std::endl;
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
    std::vector<std::string> velocity_names(dim, "v");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation_velocity(dim, DataComponentInterpretation::component_is_part_of_vector);
    u_n.update_ghost_values();
    data_out.add_data_vector(dof_handler_velocity, u_n, velocity_names, component_interpretation_velocity);
    pres_n.update_ghost_values();
    data_out.add_data_vector(dof_handler_pressure, pres_n, "p", {DataComponentInterpretation::component_is_scalar});
    data_out.build_patches();
    const std::string output = "./" + saving_dir + "/solution-" + Utilities::int_to_string(step, 5) + ".vtu";
    data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);
  }


  template<int dim>
  void NavierStokesProjection<dim>::output_errors(const unsigned int step) {
    TimerOutput::Scope t(time_table, "Output errors");

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_velocity);
    data_out.add_data_vector(H1_error_per_cell_vel, "H1_error_velocity");
    data_out.add_data_vector(L2_error_per_cell_vel, "L2_error_velocity");
    data_out.add_data_vector(Linfty_error_per_cell_vel, "Linfty_error_velocity");
    VectorTools::interpolate(dof_handler_velocity, vel_exact, u_tmp);
    u_tmp -= u_n;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    std::vector<std::string> names(dim, "Nodal_error_velocity");
    u_tmp.update_ghost_values();
    data_out.add_data_vector(dof_handler_velocity, u_tmp, names, component_interpretation);
    data_out.build_patches();
    const std::string output_vel = "./" + saving_dir + "/errors_velocity-" + Utilities::int_to_string(step, 5) + ".vtu";
    data_out.write_vtu_in_parallel(output_vel, MPI_COMM_WORLD);

    data_out.clear();

    data_out.attach_dof_handler(dof_handler_pressure);
    data_out.add_data_vector(L2_error_per_cell_pres, "L2_error_pressure");
    data_out.add_data_vector(Linfty_error_per_cell_pres, "Linfty_error_pressure");
    VectorTools::interpolate(dof_handler_pressure, pres_exact, pres_int);
    pres_int -= pres_n;
    pres_int.update_ghost_values();
    data_out.add_data_vector(dof_handler_pressure, pres_int, "Nodal_error_pressure", {DataComponentInterpretation::component_is_scalar});
    data_out.build_patches();
    const std::string output_pres = "./" + saving_dir + "/errors_pressure-" + Utilities::int_to_string(step, 5) + ".vtu";
    data_out.write_vtu_in_parallel(output_pres, MPI_COMM_WORLD);
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
    output_results(1);
    output_errors(1);
    navier_stokes_matrix.advance_p_time(dt);
    double time = t_0 + dt;
    unsigned int n = 1;
    while(std::abs(T - time) > 1e-10) {
      time += dt;
      n++;
      pcout << "Step = " << n << " Time = " << time << std::endl;
      //--- First stage of TR-BDF2
      navier_stokes_matrix.set_TR_BDF2_stage(TR_BDF2_stage);
      verbose_cout << "  Diffusion Step stage 1 " << std::endl;
      vel_exact.advance_time(gamma*dt);
      pres_exact.advance_time(gamma*dt);
      diffusion_step();
      navier_stokes_matrix.advance_p_time(gamma*dt);
      verbose_cout << "  Projection Step stage 1" << std::endl;
      project_grad(1);
      u_tmp.equ(gamma*dt, u_tmp);
      u_star += u_tmp;
      projection_step();
      verbose_cout << "  Updating the Velocity stage 1" << std::endl;
      u_n_gamma.equ(1.0, u_star);
      project_grad(2);
      u_tmp.equ(-gamma*dt, u_tmp);
      u_n_gamma += u_tmp;
      u_n_minus_1 = u_n;
      u_n = u_n_gamma;
      pres_n = pres_int;
      analyze_results();
      if(n % output_interval == 0) {
        verbose_cout << "Plotting Solution final" << std::endl;
        output_results(n);
        output_errors(n);
      }
      if(T - time < dt && T - time > 1e-10) {
        dt = T - time;
        navier_stokes_matrix.set_dt(dt);
      }
    }
    if(n % output_interval != 0) {
      verbose_cout << "Plotting Solution final" << std::endl;
      output_results(n);
      output_errors(n);
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

    NavierStokesProjection<2> test(data);
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

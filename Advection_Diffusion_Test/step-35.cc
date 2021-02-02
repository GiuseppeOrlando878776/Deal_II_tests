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
    static const unsigned int degree = 1;

    // With this class defined, we declare class that describes the boundary
    // conditions for velocity:
    template<int dim>
    class Field: public Function<dim> {
    public:
      Field(const double initial_time = 0.0);

      Field(const double Re, const double initial_time = 0.0);

      virtual double value(const Point<dim>&  p,
                           const unsigned int component = 0) const override;

      virtual Tensor<1, dim, double> gradient(const Point<dim>& p,
                                              const unsigned int component = 0) const override;

      virtual void vector_gradient(const Point<dim>& p, std::vector<Tensor<1, dim, double>>& gradients) const override;

    private:
      double Re;
    };


    template<int dim>
    Field<dim>::Field(const double initial_time): Function<dim>(1, initial_time), Re() {}


    template<int dim>
    Field<dim>::Field(const double Re, const double initial_time): Function<dim>(1, initial_time), Re(Re) {}


    template<int dim>
    double Field<dim>::value(const Point<dim>& p, const unsigned int component) const {
      (void)component;
      AssertIndexRange(component, 1);

      const double curr_time = this->get_time();

      return std::cos(p(0))*std::sin(p(1))*std::exp(-2.0*curr_time/Re);
    }


    template<int dim>
    Tensor<1, dim, double> Field<dim>::gradient(const Point<dim>& p, const unsigned int component) const {
      (void)component;
      AssertIndexRange(component, 1);

      Tensor<1, dim, double> result;
      const double curr_time = this->get_time();
      result[0] = -std::sin(p(0))*std::sin(p(1))*std::exp(-2.0*curr_time/Re);
      result[1] =  std::cos(p(0))*std::cos(p(1))*std::exp(-2.0*curr_time/Re);

      return result;
    }


    template<int dim>
    void Field<dim>::vector_gradient(const Point<dim>& p, std::vector<Tensor<1, dim, double>>& gradients) const {
      Assert(gradients.size() == dim, ExcDimensionMismatch(gradients.size(), dim));
      for(unsigned int i = 0; i < dim; ++i)
        gradients[i] = gradient(p, i);
    }


    // With this class defined, we declare class that describes the boundary
    // conditions for velocity:
    template<int dim>
    class Field_Source: public Function<dim> {
    public:
      Field_Source(const double initial_time = 0.0);

      Field_Source(const double Re, const double initial_time = 0.0);

      virtual double value(const Point<dim>&  p,
                           const unsigned int component = 0) const override;

    private:
      double Re;
    };


    template<int dim>
    Field_Source<dim>::Field_Source(const double initial_time): Function<dim>(1, initial_time), Re() {}


    template<int dim>
    Field_Source<dim>::Field_Source(const double Re, const double initial_time): Function<dim>(1, initial_time), Re(Re) {}


    template<int dim>
    double Field_Source<dim>::value(const Point<dim>& p, const unsigned int component) const {
      (void)component;
      AssertIndexRange(component, 1);

      const double curr_time = this->get_time();

      return std::cos(p(0) + p(1))*std::exp(-2.0*curr_time/Re);
    }

  } // namespace EquationData



  // @sect3{ <code>AdvectionDiffusionOperator::AdvectionDiffusionOperator</code> }
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  class AdvectionDiffusionOperator: public MatrixFreeOperators::Base<dim, Vec> {
  public:
    AdvectionDiffusionOperator();

    AdvectionDiffusionOperator(RunTimeParameters::Data_Storage& data);

    void set_dt(const double time_step);

    void set_TR_BDF2_stage(const unsigned int stage);

    void set_time(const double curr_time);

    void advance_time_source(const double time_adv);

    void vmult_rhs(Vec& dst, const std::vector<Vec>& src) const;

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

    virtual void apply_add(Vec& dst, const Vec& src) const override;

  private:
    const double a21 = 0.5;
    const double a22 = 0.5;

    const double theta_v = 0.0;
    const double C_u = 1.0*(fe_degree + 1)*(fe_degree + 1);

    EquationData::Field<dim>        exact_sol;
    EquationData::Field_Source<dim> source;

    void assemble_rhs_cell_term(const MatrixFree<dim, Number>&               data,
                                Vec&                                         dst,
                                const std::vector<Vec>&                      src,
                                const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term(const MatrixFree<dim, Number>&               data,
                                Vec&                                         dst,
                                const std::vector<Vec>&                      src,
                                const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term(const MatrixFree<dim, Number>&               data,
                                    Vec&                                         dst,
                                    const std::vector<Vec>&                      src,
                                    const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_cell_term(const MatrixFree<dim, Number>&               data,
                            Vec&                                         dst,
                            const Vec&                                   src,
                            const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_face_term(const MatrixFree<dim, Number>&               data,
                            Vec&                                         dst,
                            const Vec&                                   src,
                            const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_boundary_term(const MatrixFree<dim, Number>&               data,
                                Vec&                                         dst,
                                const Vec&                                   src,
                                const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_diagonal_cell_term(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_diagonal_face_term(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_diagonal_boundary_term(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const Vec&                                   src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const;
  };


  // Default constructor
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  AdvectionDiffusionOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::AdvectionDiffusionOperator():
    MatrixFreeOperators::Base<dim, Vec>(), Re(), dt(), gamma(), a31(), a32(), a33() {}


  // Constructor with runtime parameters storage
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  AdvectionDiffusionOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  AdvectionDiffusionOperator(RunTimeParameters::Data_Storage& data):
    MatrixFreeOperators::Base<dim, Vec>(), eq_data(data), Re(data.Reynolds), dt(5e-4),
                                           gamma(2.0 - std::sqrt(2.0)), a31((1.0 - gamma)/(2.0*(2.0 - gamma))),
                                           a32(a31), a33(1.0/(2.0 - gamma)), TR_BDF2_stage(1),
                                           exact_sol(data.Reynolds, data.initial_time),
                                           source(data.Reynolds, data.initial_time) {}

  // Setter of time-step
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void AdvectionDiffusionOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  set_dt(const double time_step) {
    dt = time_step;
  }


  // Setter of TR-BDF2 stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void AdvectionDiffusionOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  set_TR_BDF2_stage(const unsigned int stage) {
    AssertIndexRange(stage, 3);
    Assert(stage > 0, ExcInternalError());
    TR_BDF2_stage = stage;
  }


  // Setter of exact velocity for boundary conditions
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void AdvectionDiffusionOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::set_time(const double curr_time) {
    exact_sol.set_time(curr_time);
  }


  // Setter of exact velocity for boundary conditions
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void AdvectionDiffusionOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::advance_time_source(const double time_adv) {
    source.advance_time(time_adv);
  }


  // Assemble rhs cell term for the velocity
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void AdvectionDiffusionOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  assemble_rhs_cell_term(const MatrixFree<dim, Number>&               data,
                         Vec&                                         dst,
                         const std::vector<Vec>&                      src,
                         const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(TR_BDF2_stage == 1) {
      FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi(data), phi_old(data);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_old.reinit(cell);
        phi_old.gather_evaluate(src[0], true, true);
        Tensor<1, dim, VectorizedArray<Number>> adv_vel;
        for(unsigned int d = 0; d < dim; ++d)
          adv_vel[d] = make_vectorized_array<Number>(1.0);
        phi.reinit(cell);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& u_n                = phi_old.get_value(q);
          const auto& grad_u_n           = phi_old.get_gradient(q);
          const auto& tensor_product_u_n = u_n*adv_vel;
          const auto& point_vectorized   = phi.quadrature_point(q);
          auto f_old                     = VectorizedArray<Number>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d)
              point[d] = point_vectorized[d][v];
            f_old[v] = source.value(point);
          }
          phi.submit_value(1.0/(gamma*dt)*u_n + f_old, q);
          phi.submit_gradient(-a21/Re*grad_u_n + a21*tensor_product_u_n, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi(data), phi_old(data), phi_int(data);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_old.reinit(cell);
        phi_old.gather_evaluate(src[0], true, true);
        phi_int.reinit(cell);
        phi_int.gather_evaluate(src[1], true, true);
        Tensor<1, dim, VectorizedArray<Number>> adv_vel;
        for(unsigned int d = 0; d < dim; ++d)
          adv_vel[d] = make_vectorized_array<Number>(1.0);
        phi.reinit(cell);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& u_n                      = phi_old.get_value(q);
          const auto& grad_u_n                 = phi_old.get_gradient(q);
          const auto& u_n_gamma                = phi_int.get_value(q);
          const auto& grad_u_n_gamma           = phi_int.get_gradient(q);
          const auto& tensor_product_u_n       = u_n*adv_vel;
          const auto& tensor_product_u_n_gamma = u_n_gamma*adv_vel;
          const auto& point_vectorized   = phi.quadrature_point(q);
          auto f_int                     = VectorizedArray<Number>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d)
              point[d] = point_vectorized[d][v];
            f_int[v] = source.value(point);
          }
          phi.submit_value(1.0/((1.0 - gamma)*dt)*u_n_gamma + f_int, q);
          phi.submit_gradient(a32*tensor_product_u_n_gamma + a31*tensor_product_u_n -
                              a32/Re*grad_u_n_gamma - a31/Re*grad_u_n, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
  }


  // Assemble rhs face term for the velocity
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void AdvectionDiffusionOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  assemble_rhs_face_term(const MatrixFree<dim, Number>&               data,
                         Vec&                                         dst,
                         const std::vector<Vec>&                      src,
                         const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi_p(data, true), phi_m(data, false),
                                                                 phi_old_p(data, true), phi_old_m(data, false);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_old_p.reinit(face);
        phi_old_p.gather_evaluate(src[0], true, true);
        phi_old_m.reinit(face);
        phi_old_m.gather_evaluate(src[0], true, true);
        Tensor<1, dim, VectorizedArray<Number>> adv_vel;
        for(unsigned int d = 0; d < dim; ++d)
          adv_vel[d] = make_vectorized_array<Number>(1.0);
        phi_p.reinit(face);
        phi_m.reinit(face);
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus                 = phi_p.get_normal_vector(q);
          const auto& avg_grad_u_old         = 0.5*(phi_old_p.get_gradient(q) + phi_old_m.get_gradient(q));
          const auto& avg_tensor_product_u_n = 0.5*(phi_old_p.get_value(q)*adv_vel +
                                                    phi_old_m.get_value(q)*adv_vel);
          phi_p.submit_value(scalar_product(a21/Re*avg_grad_u_old - a21*avg_tensor_product_u_n, n_plus), q);
          phi_m.submit_value(-scalar_product(a21/Re*avg_grad_u_old - a21*avg_tensor_product_u_n, n_plus), q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi_p(data, true), phi_m(data, false),
                                                                 phi_old_p(data, true), phi_old_m(data, false),
                                                                 phi_int_p(data, true), phi_int_m(data, false);

      for(unsigned int face = face_range.first; face < face_range.second; ++ face) {
        phi_old_p.reinit(face);
        phi_old_p.gather_evaluate(src[0], true, true);
        phi_old_m.reinit(face);
        phi_old_m.gather_evaluate(src[0], true, true);
        phi_int_p.reinit(face);
        phi_int_p.gather_evaluate(src[1], true, true);
        phi_int_m.reinit(face);
        phi_int_m.gather_evaluate(src[1], true, true);
        Tensor<1, dim, VectorizedArray<Number>> adv_vel;
        for(unsigned int d = 0; d < dim; ++d)
          adv_vel[d] = make_vectorized_array<Number>(1.0);
        phi_p.reinit(face);
        phi_m.reinit(face);
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus                       = phi_p.get_normal_vector(q);
          const auto& avg_grad_u_old               = 0.5*(phi_old_p.get_gradient(q) + phi_old_m.get_gradient(q));
          const auto& avg_grad_u_int               = 0.5*(phi_int_p.get_gradient(q) + phi_int_m.get_gradient(q));
          const auto& avg_tensor_product_u_n       = 0.5*(phi_old_p.get_value(q)*adv_vel +
                                                          phi_old_m.get_value(q)*adv_vel);
          const auto& avg_tensor_product_u_n_gamma = 0.5*(phi_int_p.get_value(q)*adv_vel +
                                                          phi_int_m.get_value(q)*adv_vel);
          phi_p.submit_value(scalar_product(a31/Re*avg_grad_u_old + a32/Re*avg_grad_u_int -
                                            a31*avg_tensor_product_u_n - a32*avg_tensor_product_u_n_gamma , n_plus), q);
          phi_m.submit_value(-scalar_product(a31/Re*avg_grad_u_old + a32/Re*avg_grad_u_int -
                                             a31*avg_tensor_product_u_n - a32*avg_tensor_product_u_n_gamma , n_plus), q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
  }


  // Assemble rhs boundary term for the velocity
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void AdvectionDiffusionOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  assemble_rhs_boundary_term(const MatrixFree<dim, Number>&               data,
                             Vec&                                         dst,
                             const std::vector<Vec>&                      src,
                             const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi(data, true),
                                                                 phi_old(data, true);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_old.reinit(face);
        phi_old.gather_evaluate(src[0], true, true);
        Tensor<1, dim, VectorizedArray<Number>> adv_vel;
        for(unsigned int d = 0; d < dim; ++d)
          adv_vel[d] = make_vectorized_array<Number>(1.0);
        phi.reinit(face);
        const auto coef_jump = C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus             = phi.get_normal_vector(q);
          const auto& grad_u_old         = phi_old.get_gradient(q);
          const auto& tensor_product_u_n = phi_old.get_value(q)*adv_vel;
          const auto& point_vectorized   = phi.quadrature_point(q);
          auto u_int_m                   = VectorizedArray<Number>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d)
              point[d] = point_vectorized[d][v];
            u_int_m[v] = exact_sol.value(point);
          }
          const auto tensor_product_u_int_m = u_int_m*adv_vel;
          const auto lambda                 = std::abs(scalar_product(adv_vel, n_plus));
          phi.submit_value(scalar_product(a21/Re*grad_u_old - a21*tensor_product_u_n, n_plus) +
                           a22/Re*2.0*coef_jump*u_int_m -
                           a22*scalar_product(tensor_product_u_int_m, n_plus) + a22*lambda*u_int_m, q);
          phi.submit_normal_derivative(-theta_v*a22/Re*u_int_m, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi(data, true),
                                                                 phi_old(data, true),
                                                                 phi_int(data, true),
                                                                 phi_int_extr(data, true);

      for(unsigned int face = face_range.first; face < face_range.second; ++ face) {
        phi_old.reinit(face);
        phi_old.gather_evaluate(src[0], true, true);
        phi_int.reinit(face);
        phi_int.gather_evaluate(src[1], true, true);
        Tensor<1, dim, VectorizedArray<Number>> adv_vel;
        for(unsigned int d = 0; d < dim; ++d)
          adv_vel[d] = make_vectorized_array<Number>(1.0);
        phi.reinit(face);
        const auto coef_jump = C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus                   = phi.get_normal_vector(q);
          const auto& grad_u_old               = phi_old.get_gradient(q);
          const auto& grad_u_int               = phi_int.get_gradient(q);
          const auto& tensor_product_u_n       = phi_old.get_value(q)*adv_vel;
          const auto& tensor_product_u_n_gamma = phi_int.get_value(q)*adv_vel;
          const auto& point_vectorized         = phi.quadrature_point(q);
          auto u_m                             = VectorizedArray<Number>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d)
              point[d] = point_vectorized[d][v];
            u_m[v] = exact_sol.value(point);
          }
          const auto tensor_product_u_m = u_m*adv_vel;
          const auto lambda             = std::abs(scalar_product(adv_vel, n_plus));
          phi.submit_value(scalar_product(a31/Re*grad_u_old + a32/Re*grad_u_int -
                                          a31*tensor_product_u_n - a32*tensor_product_u_n_gamma, n_plus) +
                           a33/Re*2.0*coef_jump*u_m -
                           a33*scalar_product(tensor_product_u_m, n_plus) + a33*lambda*u_m, q);
          phi.submit_normal_derivative(-theta_v*a33/Re*u_m, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
  }


  // Put together all the previous steps for velocity
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void AdvectionDiffusionOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  vmult_rhs(Vec& dst, const std::vector<Vec>& src) const {
    this->data->loop(&AdvectionDiffusionOperator::assemble_rhs_cell_term,
                     &AdvectionDiffusionOperator::assemble_rhs_face_term,
                     &AdvectionDiffusionOperator::assemble_rhs_boundary_term,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble cell term for the velocity
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void AdvectionDiffusionOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  assemble_cell_term(const MatrixFree<dim, Number>&               data,
                     Vec&                                         dst,
                     const Vec&                                   src,
                     const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(TR_BDF2_stage == 1) {
      FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi(data);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi.reinit(cell);
        phi.gather_evaluate(src, true, true);
        Tensor<1, dim, VectorizedArray<Number>> adv_vel;
        for(unsigned int d = 0; d < dim; ++d)
          adv_vel[d] = make_vectorized_array<Number>(1.0);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& u_int                = phi.get_value(q);
          const auto& grad_u_int           = phi.get_gradient(q);
          const auto& tensor_product_u_int = u_int*adv_vel;
          phi.submit_value(1.0/(gamma*dt)*u_int, q);
          phi.submit_gradient(-a22*tensor_product_u_int + a22/Re*grad_u_int, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi(data);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi.reinit(cell);
        phi.gather_evaluate(src, true, true);
        Tensor<1, dim, VectorizedArray<Number>> adv_vel;
        for(unsigned int d = 0; d < dim; ++d)
          adv_vel[d] = make_vectorized_array<Number>(1.0);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& u_curr                   = phi.get_value(q);
          const auto& grad_u_curr              = phi.get_gradient(q);
          const auto& tensor_product_u_curr    = u_curr*adv_vel;
          phi.submit_value(1.0/((1.0 - gamma)*dt)*u_curr, q);
          phi.submit_gradient(-a33*tensor_product_u_curr + a33/Re*grad_u_curr, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
  }


  // Assemble face term for the velocity
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void AdvectionDiffusionOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  assemble_face_term(const MatrixFree<dim, Number>&               data,
                     Vec&                                         dst,
                     const Vec&                                   src,
                     const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi_p(data, true), phi_m(data, false);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_p.reinit(face);
        phi_p.gather_evaluate(src, true, true);
        phi_m.reinit(face);
        phi_m.gather_evaluate(src, true, true);
        const auto coef_jump = C_u*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                        std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));
        Tensor<1, dim, VectorizedArray<Number>> adv_vel;
        for(unsigned int d = 0; d < dim; ++d)
          adv_vel[d] = make_vectorized_array<Number>(1.0);
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus                   = phi_p.get_normal_vector(q);
          const auto& avg_grad_u_int           = 0.5*(phi_p.get_gradient(q) + phi_m.get_gradient(q));
          const auto& jump_u_int               = phi_p.get_value(q) - phi_m.get_value(q);
          const auto& avg_tensor_product_u_int = 0.5*(phi_p.get_value(q)*adv_vel +
                                                      phi_m.get_value(q)*adv_vel);
          const auto  lambda                   = std::max(std::abs(scalar_product(adv_vel, n_plus)),
                                                          std::abs(scalar_product(adv_vel, n_plus)));
          phi_p.submit_value(a22/Re*(-scalar_product(avg_grad_u_int, n_plus) + coef_jump*jump_u_int) +
                             a22*scalar_product(avg_tensor_product_u_int, n_plus) + 0.5*a22*lambda*jump_u_int, q);
          phi_m.submit_value(-a22/Re*(-scalar_product(avg_grad_u_int, n_plus) + coef_jump*jump_u_int) -
                              a22*scalar_product(avg_tensor_product_u_int, n_plus) - 0.5*a22*lambda*jump_u_int, q);
          phi_p.submit_normal_derivative(-theta_v*a22/Re*0.5*jump_u_int, q);
          phi_m.submit_normal_derivative(-theta_v*a22/Re*0.5*jump_u_int, q);
        }
        phi_p.integrate_scatter(true, true, dst);
        phi_m.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi_p(data, true), phi_m(data, false);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_p.reinit(face);
        phi_p.gather_evaluate(src, true, true);
        phi_m.reinit(face);
        phi_m.gather_evaluate(src, true, true);
        Tensor<1, dim, VectorizedArray<Number>> adv_vel;
        for(unsigned int d = 0; d < dim; ++d)
          adv_vel[d] = make_vectorized_array<Number>(1.0);
        const auto coef_jump = C_u*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                        std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus               = phi_p.get_normal_vector(q);
          const auto& avg_grad_u           = 0.5*(phi_p.get_gradient(q) + phi_m.get_gradient(q));
          const auto& jump_u               = phi_p.get_value(q) - phi_m.get_value(q);
          const auto& avg_tensor_product_u = 0.5*(phi_p.get_value(q)*adv_vel +
                                                  phi_m.get_value(q)*adv_vel);
          const auto  lambda               = std::max(std::abs(scalar_product(adv_vel, n_plus)),
                                                      std::abs(scalar_product(adv_vel, n_plus)));
          phi_p.submit_value(a33/Re*(-scalar_product(avg_grad_u, n_plus) + coef_jump*jump_u) +
                             a33*scalar_product(avg_tensor_product_u, n_plus) + 0.5*a33*lambda*jump_u, q);
          phi_m.submit_value(-a33/Re*(-scalar_product(avg_grad_u, n_plus) + coef_jump*jump_u) -
                              a33*scalar_product(avg_tensor_product_u, n_plus) - 0.5*a33*lambda*jump_u, q);
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
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void AdvectionDiffusionOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  assemble_boundary_term(const MatrixFree<dim, Number>&               data,
                         Vec&                                         dst,
                         const Vec&                                   src,
                         const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi(data, true);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi.reinit(face);
        phi.gather_evaluate(src, true, true);
        const auto coef_jump  = C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);
        const double coef_trasp = 0.0;
        Tensor<1, dim, VectorizedArray<Number>> adv_vel;
        for(unsigned int d = 0; d < dim; ++d)
          adv_vel[d] = make_vectorized_array<Number>(1.0);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus               = phi.get_normal_vector(q);
          const auto& tensor_product_n     = outer_product(n_plus, n_plus);
          const auto& grad_u_int           = phi.get_gradient(q);
          const auto& u_int                = phi.get_value(q);
          const auto& tensor_product_u_int = phi.get_value(q)*adv_vel;
          const auto  lambda               = std::abs(scalar_product(adv_vel, n_plus));
          phi.submit_value(a22/Re*(-scalar_product(grad_u_int, n_plus) + 2.0*coef_jump*u_int) +
                           a22*coef_trasp*scalar_product(tensor_product_u_int, n_plus) + a22*lambda*u_int, q);
          phi.submit_normal_derivative(-theta_v*a22/Re*u_int, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi(data, true);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi.reinit(face);
        phi.gather_evaluate(src, true, true);
        const auto coef_jump  = C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);
        const double coef_trasp = 0.0;
        Tensor<1, dim, VectorizedArray<Number>> adv_vel;
        for(unsigned int d = 0; d < dim; ++d)
          adv_vel[d] = make_vectorized_array<Number>(1.0);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus           = phi.get_normal_vector(q);
          const auto& tensor_product_n = outer_product(n_plus, n_plus);
          const auto& grad_u           = phi.get_gradient(q);
          const auto& u                = phi.get_value(q);
          const auto& tensor_product_u = phi.get_value(q)*adv_vel;
          const auto  lambda           = std::abs(scalar_product(adv_vel, n_plus));
          phi.submit_value(a33/Re*(-scalar_product(grad_u,n_plus) + 2.0*coef_jump*u) +
                           a33*coef_trasp*scalar_product(tensor_product_u,n_plus) + a33*lambda*u, q);
          phi.submit_normal_derivative(-theta_v*a33/Re*u, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
  }


  // Put together all previous steps
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void AdvectionDiffusionOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  apply_add(Vec& dst, const Vec& src) const {
    this->data->loop(&AdvectionDiffusionOperator::assemble_cell_term,
                     &AdvectionDiffusionOperator::assemble_face_term,
                     &AdvectionDiffusionOperator::assemble_boundary_term,
                     this, dst, src, false,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble diagonal cell term for the velocity
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void AdvectionDiffusionOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  assemble_diagonal_cell_term(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(TR_BDF2_stage == 1) {
      FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi(data);

      AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);
      VectorizedArray<Number> tmp;
      tmp = make_vectorized_array<Number>(1.0);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        Tensor<1, dim, VectorizedArray<Number>> adv_vel;
        for(unsigned int d = 0; d < dim; ++d)
          adv_vel[d] = make_vectorized_array<Number>(1.0);
        phi.reinit(cell);
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
            phi.submit_dof_value(VectorizedArray<Number>(), j);
          phi.submit_dof_value(tmp, i);
          phi.evaluate(true, true);
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& u_int                = phi.get_value(q);
            const auto& grad_u_int           = phi.get_gradient(q);
            const auto& tensor_product_u_int = u_int*adv_vel;
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
      FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi(data);

      AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);
      VectorizedArray<Number> tmp;
      tmp = make_vectorized_array<Number>(1.0);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        Tensor<1, dim, VectorizedArray<Number>> adv_vel;
        for(unsigned int d = 0; d < dim; ++d)
          adv_vel[d] = make_vectorized_array<Number>(1.0);
        phi.reinit(cell);
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
            phi.submit_dof_value(VectorizedArray<Number>(), j);
          phi.submit_dof_value(tmp, i);
          phi.evaluate(true, true);
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& u_curr                   = phi.get_value(q);
            const auto& grad_u_curr              = phi.get_gradient(q);
            const auto& tensor_product_u_curr    = u_curr*adv_vel;
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
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void AdvectionDiffusionOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  assemble_diagonal_face_term(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi_p(data, true), phi_m(data, false);

      AssertDimension(phi_p.dofs_per_component, phi_m.dofs_per_component);
      AlignedVector<VectorizedArray<Number>> diagonal_p(phi_p.dofs_per_component), diagonal_m(phi_m.dofs_per_component);
      VectorizedArray<Number> tmp;
      tmp = make_vectorized_array<Number>(1.0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        Tensor<1, dim, VectorizedArray<Number>> adv_vel;
        for(unsigned int d = 0; d < dim; ++d)
          adv_vel[d] = make_vectorized_array<Number>(1.0);
        phi_p.reinit(face);
        phi_m.reinit(face);
        const auto coef_jump = C_u*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                        std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));
        for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi_p.dofs_per_component; ++j) {
            phi_p.submit_dof_value(VectorizedArray<Number>(), j);
            phi_m.submit_dof_value(VectorizedArray<Number>(), j);
          }
          phi_p.submit_dof_value(tmp, i);
          phi_p.evaluate(true, true);
          phi_m.submit_dof_value(tmp, i);
          phi_m.evaluate(true, true);
          for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
            const auto& n_plus                   = phi_p.get_normal_vector(q);
            const auto& avg_grad_u_int           = 0.5*(phi_p.get_gradient(q) + phi_m.get_gradient(q));
            const auto& jump_u_int               = phi_p.get_value(q) - phi_m.get_value(q);
            const auto& avg_tensor_product_u_int = 0.5*(phi_p.get_value(q)*adv_vel +
                                                        phi_m.get_value(q)*adv_vel);
            const auto  lambda                   = std::max(std::abs(scalar_product(adv_vel, n_plus)),
                                                            std::abs(scalar_product(adv_vel, n_plus)));
            phi_p.submit_value(a22/Re*(-scalar_product(avg_grad_u_int, n_plus) + coef_jump*jump_u_int) +
                               a22*scalar_product(avg_tensor_product_u_int, n_plus) + 0.5*a22*lambda*jump_u_int , q);
            phi_m.submit_value(-a22/Re*(-scalar_product(avg_grad_u_int, n_plus) + coef_jump*jump_u_int) -
                               a22*scalar_product(avg_tensor_product_u_int, n_plus) - 0.5*a22*lambda*jump_u_int, q);
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
      FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi_p(data, true), phi_m(data, false);

      AssertDimension(phi_p.dofs_per_component, phi_m.dofs_per_component);
      AlignedVector<VectorizedArray<Number>> diagonal_p(phi_p.dofs_per_component), diagonal_m(phi_m.dofs_per_component);
      VectorizedArray<Number> tmp;
      tmp = make_vectorized_array<Number>(1.0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        Tensor<1, dim, VectorizedArray<Number>> adv_vel;
        for(unsigned int d = 0; d < dim; ++d)
          adv_vel[d] = make_vectorized_array<Number>(1.0);
        phi_p.reinit(face);
        phi_m.reinit(face);
        const auto coef_jump = C_u*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                        std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));
        for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi_p.dofs_per_component; ++j) {
            phi_p.submit_dof_value(VectorizedArray<Number>(), j);
            phi_m.submit_dof_value(VectorizedArray<Number>(), j);
          }
          phi_p.submit_dof_value(tmp, i);
          phi_p.evaluate(true, true);
          phi_m.submit_dof_value(tmp, i);
          phi_m.evaluate(true, true);
          for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
            const auto& n_plus               = phi_p.get_normal_vector(q);
            const auto& avg_grad_u           = 0.5*(phi_p.get_gradient(q) + phi_m.get_gradient(q));
            const auto& jump_u               = phi_p.get_value(q) - phi_m.get_value(q);
            const auto& avg_tensor_product_u = 0.5*(phi_p.get_value(q)*adv_vel +
                                                    phi_m.get_value(q)*adv_vel);
            const auto  lambda               = std::max(std::abs(scalar_product(adv_vel, n_plus)),
                                                        std::abs(scalar_product(adv_vel, n_plus)));
            phi_p.submit_value(a33/Re*(-scalar_product(avg_grad_u, n_plus) + coef_jump*jump_u) +
                               a33*scalar_product(avg_tensor_product_u, n_plus) + 0.5*a33*lambda*jump_u, q);
            phi_m.submit_value(-a33/Re*(-scalar_product(avg_grad_u, n_plus) + coef_jump*jump_u) -
                               a33*scalar_product(avg_tensor_product_u, n_plus) - 0.5*a33*lambda*jump_u, q);
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
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void AdvectionDiffusionOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::
  assemble_diagonal_boundary_term(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const Vec&                                   src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi(data, true);

      AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);
      VectorizedArray<Number> tmp;
      tmp = make_vectorized_array<Number>(1.0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        Tensor<1, dim, VectorizedArray<Number>> adv_vel;
        for(unsigned int d = 0; d < dim; ++d)
          adv_vel[d] = make_vectorized_array<Number>(1.0);
        phi.reinit(face);
        const auto coef_jump    = C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);
        const double coef_trasp = 0.0;
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
            phi.submit_dof_value(VectorizedArray<Number>(), j);
          phi.submit_dof_value(tmp, i);
          phi.evaluate(true, true);
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& n_plus               = phi.get_normal_vector(q);
            const auto& tensor_product_n     = outer_product(n_plus, n_plus);
            const auto& grad_u_int           = phi.get_gradient(q);
            const auto& u_int                = phi.get_value(q);
            const auto& tensor_product_u_int = phi.get_value(q)*adv_vel;
            const auto  lambda               = std::abs(scalar_product(adv_vel, n_plus));
            phi.submit_value(a22/Re*(-scalar_product(grad_u_int, n_plus) + 2.0*coef_jump*u_int) +
                             a22*coef_trasp*scalar_product(tensor_product_u_int, n_plus) + a22*lambda*u_int, q);
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
      FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi(data, true);

      AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);
      VectorizedArray<Number> tmp;
      tmp = make_vectorized_array<Number>(1.0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        Tensor<1, dim, VectorizedArray<Number>> adv_vel;
        for(unsigned int d = 0; d < dim; ++d)
          adv_vel[d] = make_vectorized_array<Number>(1.0);
        phi.reinit(face);
        const auto coef_jump    = C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);
        const double coef_trasp = 0.0;
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
            phi.submit_dof_value(VectorizedArray<Number>(), j);
          phi.submit_dof_value(tmp, i);
          phi.evaluate(true, true);
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& n_plus           = phi.get_normal_vector(q);
            const auto& tensor_product_n = outer_product(n_plus, n_plus);
            const auto& grad_u           = phi.get_gradient(q);
            const auto& u                = phi.get_value(q);
            const auto& tensor_product_u = phi.get_value(q)*adv_vel;
            const auto  lambda           = std::abs(scalar_product(adv_vel, n_plus));
            phi.submit_value(a33/Re*(-scalar_product(grad_u, n_plus) + coef_jump*u) +
                             a33*coef_trasp*scalar_product(tensor_product_u, n_plus) + a33*lambda*u, q);
            phi.submit_normal_derivative(-theta_v*a33/Re*u, q);
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


  // Put together all previous steps
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec, typename Number>
  void AdvectionDiffusionOperator<dim, fe_degree, n_q_points_1d, Vec, Number>::compute_diagonal() {
    this->inverse_diagonal_entries.reset(new DiagonalMatrix<Vec>());
    auto& inverse_diagonal = this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal);
    Vec dummy;
    dummy.reinit(inverse_diagonal.local_size());
    this->data->loop(&AdvectionDiffusionOperator::assemble_diagonal_cell_term,
                     &AdvectionDiffusionOperator::assemble_diagonal_face_term,
                     &AdvectionDiffusionOperator::assemble_diagonal_boundary_term,
                     this, inverse_diagonal, dummy, false,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
    for(unsigned int i = 0; i < inverse_diagonal.local_size(); ++i) {
      Assert(inverse_diagonal.local_element(i) != 0.0,
             ExcMessage("No diagonal entry in a definite operator should be zero"));
      inverse_diagonal.local_element(i) = 1.0/inverse_diagonal.local_element(i);
    }
  }



  // @sect3{The <code>AdvectionDiffusion</code> class}

  // Now for the main class of the program. It implements the various versions
  // of the projection method for Navier-Stokes equations.
  //
  template<int dim>
  class AdvectionDiffusion {
  public:
    AdvectionDiffusion(RunTimeParameters::Data_Storage& data);

    void run(const bool verbose = false, const unsigned int output_interval = 10);

  protected:
    const double       t_0;
    const double       T;
    const double       gamma;         //--- TR-BDF2 parameter
    unsigned int       TR_BDF2_stage; //--- Flag to check at which current stage of TR-BDF2 are
    const double       Re;
    const double       CFL;

    EquationData::Field<dim> exact_sol;

    Triangulation<dim> triangulation;

    FESystem<dim> fe;

    DoFHandler<dim> dof_handler;

    QGauss<dim> quadrature;

    LinearAlgebra::distributed::Vector<double> u_n;
    LinearAlgebra::distributed::Vector<double> u_n_minus_1;
    LinearAlgebra::distributed::Vector<double> u_n_gamma;
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

    void solve();

    void analyze_results();

    void output_results(const unsigned int step);

    void output_errors(const unsigned int step);

  private:
    std::shared_ptr<MatrixFree<dim, double>> matrix_free_storage;

    AdvectionDiffusionOperator<dim, EquationData::degree, EquationData::degree + 1,
                               LinearAlgebra::distributed::Vector<double>, double> navier_stokes_matrix;

    AffineConstraints<double> constraint;

    unsigned int vel_max_its;
    unsigned int vel_Krylov_size;
    unsigned int vel_off_diagonals;
    unsigned int vel_update_prec;
    double       vel_eps;
    double       vel_diag_strength;

    Vector<double> H1_error_per_cell, L2_error_per_cell, Linfty_error_per_cell,
                   H1_rel_error_per_cell, L2_rel_error_per_cell, Linfty_rel_error_per_cell,
                   Nodal_error_per_cell;

    double dt;

    std::string saving_dir;

    std::ofstream output_error;
  };


  // @sect4{ <code>AdvectionDiffusion::AdvectionDiffusion</code> }

  // In the constructor, we just read all the data from the
  // <code>Data_Storage</code> object that is passed as an argument, verify that
  // the data we read are reasonable and, finally, create the triangulation and
  // load the initial data.
  template<int dim>
  AdvectionDiffusion<dim>::AdvectionDiffusion(RunTimeParameters::Data_Storage& data):
    t_0(data.initial_time),
    T(data.final_time),
    gamma(2.0 - std::sqrt(2.0)),  //--- Save also in the NavierStokes class the TR-BDF2 parameter value
    TR_BDF2_stage(1),             //--- Initialize the flag for the TR_BDF2 stage
    Re(data.Reynolds),
    CFL(data.CFL),
    exact_sol(data.Reynolds, data.initial_time),
    fe(FE_DGQ<dim>(EquationData::degree), 1),
    dof_handler(triangulation),
    quadrature(EquationData::degree + 1),
    navier_stokes_matrix(data),
    vel_max_its(data.vel_max_iterations),
    vel_Krylov_size(data.vel_Krylov_size),
    vel_off_diagonals(data.vel_off_diagonals),
    vel_update_prec(data.vel_update_prec),
    vel_eps(data.vel_eps),
    vel_diag_strength(data.vel_diag_strength),
    saving_dir(data.dir),
    output_error("./" + data.dir + "/error_analysis.dat", std::ofstream::out) {
      if(EquationData::degree < 1) {
        std::cout
        << " WARNING: The chosen finite element spaces is not stable."
        << std::endl
        << " The obtained results will be nonsense" << std::endl;
      }

      AssertThrow(!((dt <= 0.0) || (dt > 0.5*T)), ExcInvalidTimeStep(dt, 0.5*T));

      constraint.clear();
      create_triangulation_and_dofs(data.n_global_refines);
      initialize();
  }


  // @sect4{<code>AdvectionDiffusion::create_triangulation_and_dofs</code>}

  // The method that creates the triangulation and refines it the needed number
  // of times. After creating the triangulation, it creates the mesh dependent
  // data, i.e. it distributes degrees of freedom and renumbers them, and
  // initializes the matrices and vectors that we will use.
  //
  template<int dim>
  void AdvectionDiffusion<dim>::create_triangulation_and_dofs(const unsigned int n_refines) {
    Point<dim> upper_right;
    upper_right[0] = 2.0*numbers::PI;
    for(unsigned int d = 1; d < dim; ++d)
      upper_right[d] = upper_right[0];
    GridGenerator::hyper_rectangle(triangulation, Point<dim>(), upper_right, true);

    std::cout << "Number of refines = " << n_refines << std::endl;
    triangulation.refine_global(n_refines);
    std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
    dt = CFL*2.0*numbers::PI/(std::pow(2, n_refines))/(EquationData::degree);
    navier_stokes_matrix.set_dt(dt);

    Vector<double> error_per_cell_tmp(triangulation.n_active_cells());
    H1_error_per_cell.reinit(error_per_cell_tmp);
    L2_error_per_cell.reinit(error_per_cell_tmp);
    Linfty_error_per_cell.reinit(error_per_cell_tmp);
    H1_rel_error_per_cell.reinit(error_per_cell_tmp);
    L2_rel_error_per_cell.reinit(error_per_cell_tmp);
    Linfty_rel_error_per_cell.reinit(error_per_cell_tmp);

    dof_handler.distribute_dofs(fe);

    Nodal_error_per_cell.reinit(dof_handler.n_dofs());

    std::cout << "dim (X_h) = " << dof_handler.n_dofs()
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
    dof_handlers.push_back(&dof_handler);

    std::vector<const AffineConstraints<double>*> constraints;
    constraints.push_back(&constraint);

    std::vector<QGauss<dim - 1>> quadratures;
    quadratures.push_back(QGauss<dim - 1>(EquationData::degree + 1));

    matrix_free_storage = std::make_unique<MatrixFree<dim, double>>();
    matrix_free_storage->reinit(dof_handlers, constraints, quadratures, additional_data);
    matrix_free_storage->initialize_dof_vector(rhs_u);
    matrix_free_storage->initialize_dof_vector(u_n);
    matrix_free_storage->initialize_dof_vector(u_n_minus_1);
    matrix_free_storage->initialize_dof_vector(u_n_gamma);
    matrix_free_storage->initialize_dof_vector(u_tmp);
  }


  // @sect4{ <code>AdvectionDiffusion::initialize</code> }

  // This method loads the initial data
  //
  template<int dim>
  void AdvectionDiffusion<dim>::initialize() {
    exact_sol.set_time(t_0);
    VectorTools::interpolate(dof_handler, exact_sol, u_n_minus_1);
    exact_sol.advance_time(dt);
    VectorTools::interpolate(dof_handler, exact_sol, u_n);
  }


  // @sect4{<code>AdvectionDiffusion::diffusion_step</code>}

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
  void AdvectionDiffusion<dim>::solve() {
    navier_stokes_matrix.initialize(matrix_free_storage);
    navier_stokes_matrix.set_time(exact_sol.get_time());
    if(TR_BDF2_stage == 1)
      navier_stokes_matrix.vmult_rhs(rhs_u, {u_n});
    else
      navier_stokes_matrix.vmult_rhs(rhs_u, {u_n, u_n_gamma});

    SolverControl solver_control(vel_max_its, vel_eps*rhs_u.l2_norm());
    SolverGMRES<LinearAlgebra::distributed::Vector<double>>
    gmres(solver_control, SolverGMRES<LinearAlgebra::distributed::Vector<double>>::AdditionalData(vel_Krylov_size));
    PreconditionJacobi<AdvectionDiffusionOperator<dim,
                                                      EquationData::degree,
                                                      EquationData::degree + 1,
                                                      LinearAlgebra::distributed::Vector<double>,
                                                      double>> preconditioner;
    navier_stokes_matrix.compute_diagonal();
    preconditioner.initialize(navier_stokes_matrix);
    if(TR_BDF2_stage == 1)
      gmres.solve(navier_stokes_matrix, u_n_gamma, rhs_u, preconditioner);
    else
      gmres.solve(navier_stokes_matrix, u_n, rhs_u, preconditioner);
  }


  // Since we have solved a problem with analytic solution, we want to verify
  // the correctness of our implementation by computing the errors of the
  // numerical result against the analytic solution.
  //
  template <int dim>
  void AdvectionDiffusion<dim>::analyze_results() {
    u_tmp = 0;

    VectorTools::integrate_difference(dof_handler, u_n, exact_sol,
                                      H1_error_per_cell, quadrature, VectorTools::H1_norm);
    const double error_H1 = VectorTools::compute_global_error(triangulation, H1_error_per_cell, VectorTools::H1_norm);
    VectorTools::integrate_difference(dof_handler, u_tmp, exact_sol,
                                      H1_rel_error_per_cell, quadrature, VectorTools::H1_norm);
    const double H1 = VectorTools::compute_global_error(triangulation, H1_rel_error_per_cell, VectorTools::H1_norm);
    for(unsigned int d = 0; d < triangulation.n_active_cells(); ++d)
      H1_rel_error_per_cell[d] = H1_error_per_cell[d]/H1_rel_error_per_cell[d];
    const double error_rel_H1 = error_H1/H1;
    std::cout << "Verification via H1 error:    "<< error_H1 << std::endl;
    std::cout << "Verification via H1 relative error:    "<< error_rel_H1 << std::endl;

    VectorTools::integrate_difference(dof_handler, u_n, exact_sol,
                                      L2_error_per_cell, quadrature, VectorTools::L2_norm);
    const double error_L2 = VectorTools::compute_global_error(triangulation, L2_error_per_cell, VectorTools::L2_norm);
    VectorTools::integrate_difference(dof_handler, u_tmp, exact_sol,
                                      L2_rel_error_per_cell, quadrature, VectorTools::L2_norm);
    const double L2 = VectorTools::compute_global_error(triangulation, L2_rel_error_per_cell, VectorTools::L2_norm);
    for(unsigned int d = 0; d < triangulation.n_active_cells(); ++d)
      L2_rel_error_per_cell[d] = L2_error_per_cell[d]/L2_rel_error_per_cell[d];
    const double error_rel_L2 = error_L2/L2;
    std::cout << "Verification via L2 error:    "<< error_L2 << std::endl;
    std::cout << "Verification via L2 relative error:    "<< error_rel_L2 << std::endl;

    VectorTools::integrate_difference(dof_handler, u_n, exact_sol,
                                      Linfty_error_per_cell, quadrature, VectorTools::Linfty_norm);
    const double error_Linfty = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell, VectorTools::Linfty_norm);
    VectorTools::integrate_difference(dof_handler, u_tmp, exact_sol,
                                      Linfty_rel_error_per_cell, quadrature, VectorTools::Linfty_norm);
    const double Linfty = VectorTools::compute_global_error(triangulation, Linfty_rel_error_per_cell, VectorTools::Linfty_norm);
    for(unsigned int d = 0; d < triangulation.n_active_cells(); ++d)
      Linfty_rel_error_per_cell[d] = Linfty_error_per_cell[d]/Linfty_rel_error_per_cell[d];
    const double error_rel_Linfty = error_Linfty/Linfty;
    std::cout << "Verification via Linfty error:    "<< error_Linfty << std::endl;
    std::cout << "Verification via Linfty relative error:    "<< error_rel_Linfty << std::endl;

    output_error << error_H1 << std::endl;
    output_error << error_rel_H1 << std::endl;
    output_error << error_L2 << std::endl;
    output_error << error_rel_L2 << std::endl;
    output_error << error_Linfty << std::endl;
    output_error << error_rel_Linfty << std::endl;
  }


  // @sect4{ <code>AdvectionDiffusion::output_results</code> }

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
  void AdvectionDiffusion<dim>::output_results(const unsigned int step) {
  }


  template<int dim>
  void AdvectionDiffusion<dim>::output_errors(const unsigned int step) {
  }


  // @sect4{ <code>AdvectionDiffusion::run</code> }

  // This is the time marching function, which starting at <code>t_0</code>
  // advances in time using the projection method with time step <code>dt</code>
  // until <code>T</code>.
  //
  // Its second parameter, <code>verbose</code> indicates whether the function
  // should output information what it is doing at any given moment:
  // we use the ConditionalOStream class to do that for us.
  //
  template<int dim>
  void AdvectionDiffusion<dim>::run(const bool verbose, const unsigned int output_interval) {
    ConditionalOStream verbose_cout(std::cout, verbose);

    analyze_results();
    output_results(1);
    output_errors(1);
    navier_stokes_matrix.advance_time_source(dt);
    double time = t_0 + dt;
    unsigned int n = 1;
    while(std::abs(T - time) > 1e-10) {
      time += dt;
      n++;
      std::cout << "Step = " << n << " Time = " << time << std::endl;
      //--- First stage of TR-BDF2
      verbose_cout << "  Solution stage 1 " << std::endl;
      exact_sol.advance_time(gamma*dt);
      navier_stokes_matrix.set_TR_BDF2_stage(TR_BDF2_stage);
      solve();
      navier_stokes_matrix.advance_time_source(gamma*dt);
      u_n_minus_1 = u_n;
      TR_BDF2_stage = 2; //--- Flag to pass at second stage
      //--- Second stage of TR-BDF2
      verbose_cout << "  Solution stage 2 " << std::endl;
      exact_sol.advance_time((1.0 - gamma)*dt);
      navier_stokes_matrix.set_TR_BDF2_stage(TR_BDF2_stage);
      solve();
      navier_stokes_matrix.advance_time_source((1.0 - gamma)*dt);
      TR_BDF2_stage = 1; //--- Flag to pass at first stage at next step
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

    AdvectionDiffusion<2> test(data);
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

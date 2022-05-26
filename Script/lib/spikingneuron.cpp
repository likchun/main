#include <boost/random.hpp>
#include <vector>

// g++ -shared -o spikingneuron.so -fPIC spikingneuron.cpp -I C:/Library/C++/boost_1_78_0

extern "C"
{
    static double membrane_potential;
    static double recovery_variable;
    static double current_drive;
    static double current_noise;
    static std::vector<int> spike_timesteps;

    static double a, b, c, d;

    static double delta_time;
    static double simulation_time;

    static int now_step;
    static double sqrt_dt;
    static double v_temp;
    static boost::random::mt19937 random_generator;
    static boost::random::normal_distribution<double> norm_dist;


    void parameters(
        double _delta_time,
        double _a,
        double _b,
        double _c,
        double _d,
        double _current_drive,
        double _whitenoise_standarddeviation,
        double _random_number_generation_seed,
        double _initial_membrane_potential,
        double _initial_recovery_variable
    )
    {
        a = _a;
        b = _b;
        c = _c;
        d = _d;
        now_step = 0;
        delta_time = _delta_time;
        sqrt_dt = sqrt(_delta_time);
        current_drive = _current_drive;
        random_generator = boost::random::mt19937(_random_number_generation_seed);
        norm_dist = boost::random::normal_distribution<double>(0, _whitenoise_standarddeviation);
        membrane_potential = _initial_membrane_potential;
        recovery_variable = _initial_recovery_variable;
    }


    void step(int number_of_step=1)
    {
        for (int i = 0; i < number_of_step; ++i)
        {
            ++now_step;

            v_temp = membrane_potential;    // so that the other variables take
                                            // the membrane potential of the previous step

            current_noise = norm_dist(random_generator);

            membrane_potential += (
                (0.04 * membrane_potential * membrane_potential) + (5 * membrane_potential)
                + 140 - recovery_variable + current_drive
            ) * delta_time + current_noise * sqrt_dt;

            recovery_variable += a * (
                b * v_temp - recovery_variable
            ) * delta_time;

            if (membrane_potential >= 30) {
                membrane_potential = c;
                recovery_variable += d;
                spike_timesteps.push_back(now_step);
            }
        }
    }

    double get_membrane_potential(void) { return membrane_potential; }
    double get_membrane_potential_stochastic(void) { return current_noise*sqrt_dt; }
    double get_recovery_variable(void) { return recovery_variable; }
    double get_current(void) { return current_drive+current_noise; }
    double get_current_driving(void) { return current_drive; }
    void set_current_driving(double _current_drive) { current_drive = _current_drive; }
    double get_current_stochastic(void) { return current_noise; }
}
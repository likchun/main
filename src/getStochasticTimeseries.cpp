/**
 * @file getStochasticTimeseries.cpp
 * @author likchun@outlook.com
 * @brief 
 * @version 0.1
 * @date 2022-06-13
 * 
 * @copyright free to use
 * 
 * @note to be compiled in C++ version 11 or later with boost library 1.78.0
 * 
 */

/**
 * change log
 * ver 1.0 - init
 * ver 1.1 - IO update: exported files in /output
 *         - Output update: able to export also recovery variable and current time series
 *         - Input update: synaptic weights matrix nonzero format add network size at line 1
 * 
 */


#include <boost/random.hpp>
#include <fstream>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#elif __linux__
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define _CRT_SECURE_NO_WARNINGS

#define NO_MAIN_ARGUMENT                        argc == 1
#define DEFAULT_OUTPUT_FOLDER                   "output"

#ifdef _WIN32
#define DEFAULT_OUTPUT_FILENAME_NOISE_SERIES    "output\\stoc.bin"
#define CREATE_OUTPUT_DIRECTORY(__DIR)          if (CreateDirectoryA(__DIR, NULL) || ERROR_ALREADY_EXISTS == GetLastError()) {}\
                                                else { error_handler::throw_error("dir_create", __DIR); }
#else
struct  stat st = {0};
#define DEFAULT_OUTPUT_FILENAME_NOISE_SERIES    "output/stoc.bin"
#define CREATE_OUTPUT_DIRECTORY(__DIR)          if (stat(__DIR, &st) == -1) { mkdir(__DIR, 0700); }
#endif

std::string code_ver = "Version 0.1\nBuild 1\nLast Update 13 June 2022";


namespace datatype_precision
{
    constexpr const int	DIGIT_FLOAT	 = 9;	// use SINGLE floating point precision for time series output
    constexpr const int	DIGIT_DOUBLE = 17;	// use DOUBLE floating point precision for time series output
    const double		PRECISION_FLOAT	 = pow(10, DIGIT_FLOAT);
    const double		PRECISION_DOUBLE = pow(10, DIGIT_DOUBLE);
}; // datatype_precision

namespace error_handler
{
    void throw_warning(std::string _warn, std::string _e, float _eno)
    {
        std::string msg;
        if (_warn == "param_value") {
            msg = "<invalid parameter value>: \""+_e
                +"\" is given an invalid value of "
                +std::to_string(static_cast<long long>(_eno));
        } else {
            msg = "<unknown>: warnf1";
        }
        std::cerr << "\nWarning" << msg << std::endl;
    }

    void throw_error(std::string _err, std::string _e)
    {
        std::string msg;
        if (_err == "file_access") {
            msg = "<file not found>: \""+_e+"\" cannot be found or accessed";
        } else if (_err == "dir_create") {
            msg = "<io>: directroy \""+_e+"\" cannot be created";
        } else {
            msg = "<unknown failure>: errf1";
        }
        std::cerr << "\nError" << msg << std::endl;
        exit(1);
    }

    template<typename T>
    void throw_error(std::string _err, std::vector<T> &&_e)
    {
        std::string msg;
        if (_err == "neuron_type") {
            msg = "<neuron classification>: inconsistent neuron type is detected at ("
                +std::to_string(static_cast<long long>(_e[0]))+", "
                +std::to_string(static_cast<long long>(_e[1]))+
                ") of the coupling strength matrix";
        } else {
            msg = "<unknown failure>: errf2";
        }
        std::cerr << "\nError" << msg << std::endl;
        exit(1);
    }
}; // error_handler

namespace tools
{
    std::string remove_whitespace(std::string &str)
    {
        str.erase(remove_if(str.begin(), str.end(), ::isspace), str.end());
        return str;
    }

    std::vector<int> string_to_vector_index(std::string &string_of_values, char delimiter)
    {
        std::vector<int> res;
        std::string value;
        std::stringstream ss(string_of_values);
        while (getline(ss, value, delimiter)) {
            res.push_back(std::stoi(tools::remove_whitespace(value))-1); }
        return res;
    }

    template<typename T>
    void export_array2D(std::vector<std::vector<T>> &array2D, std::string &&filename,
                        char delimiter)
    {
        std::ofstream ofs;
        ofs.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        try {
            ofs.open(filename, std::ios::trunc);
            ofs << array2D[0].size();
            for (auto &elem : array2D[0]) { ofs << delimiter << elem; }
            for (size_t row = 1; row < array2D.size(); ++row) {
                ofs << '\n' << array2D[row].size();
                for (auto &elem : array2D[row]) { ofs << delimiter << elem; }
            }
            ofs.close();
        } catch(std::ofstream::failure const&) {
            error_handler::throw_error("file_access", filename);
        }
    }
}; // tools

namespace constants
{
    constexpr const unsigned int TIMESERIES_BUFFSIZE_THRESHOLD = 150000000;
}; // constants


/* Auxiliary Functions */
void initialize_containers(
    const int network_size,
    int &now_step,
    double seed_to_random_number_generation,
    double noise_sigma,
    boost::random::mt19937 &random_generator,
    boost::random::normal_distribution<double> &norm_dist,
    std::string outfile_stochastic_timeseries,
    std::vector<float> &stochastic_timeseries_buffer,
    std::ofstream &ofs_stochastic_timeseries
)
{
    now_step = 0;
    ofs_stochastic_timeseries.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    try {
        ofs_stochastic_timeseries.open(outfile_stochastic_timeseries, std::ios::trunc | std::ios::binary);
        ofs_stochastic_timeseries.close();
        ofs_stochastic_timeseries.open(outfile_stochastic_timeseries, std::ios::app | std::ios::binary);
    } catch(std::ifstream::failure const&) {
        error_handler::throw_error("file_access", outfile_stochastic_timeseries);
    }
    stochastic_timeseries_buffer.reserve(
        static_cast<unsigned int>(
            (constants::TIMESERIES_BUFFSIZE_THRESHOLD / network_size + 1) * network_size
    ));
    for (int i = 0; i < network_size; ++i) {
        stochastic_timeseries_buffer.push_back(0.0);
    }
    boost::random::mt19937 _random_generator(seed_to_random_number_generation);
    random_generator = _random_generator;
    boost::random::normal_distribution<double> _norm_dist(0, noise_sigma);
    norm_dist = _norm_dist;
}


int main(int argc, char **argv)
{
    // int     mode = 0;			// 0: overwrite mode | 1: continue mode
    // int continuation = -1;	// count the number of times of continuation
    int	network_size, now_step, total_step;
    double sqrt_dt, seed_to_random_number_generation, noise_sigma;
    std::vector<float> stochastic_timeseries_buffer;
    std::ofstream ofs_stochastic_timeseries;
    boost::random::mt19937 random_generator;
    boost::random::normal_distribution<double> norm_dist;

    if (argc == 6) {
        network_size = std::stoi(argv[1]);
        sqrt_dt = sqrt(std::stod(argv[2]));
        total_step = (int)(std::stod(argv[3]) / std::stod(argv[2]));
        seed_to_random_number_generation = std::stod(argv[4]);
        noise_sigma = std::stod(argv[5]);
    } else {
        std::cout << "\nTo generate a time series with stochastic noise.\n\n";
        std::cout << "Usage:\n";
        std::cout << ">> ./getStochasticTimeseries [network_size] [delta_time] [simulation_time] [random_number_generation_seed] [white_noise_SD]\n\n";
        std::cout << "Example:\n";
        std::cout << ">> ./getStochasticTimeseries 4095 0.05 10000 0 3\n\n";
        exit(0);
    }

    std::cout << "Program started\n"
              << "------------------------\n"
              << "network size:     " << network_size << '\n'
              << "delta time:       " << std::stod(argv[2]) << '\n'
              << "simulation time:  " << std::stod(argv[3]) << '\n'
              << "random gen seed:  " << seed_to_random_number_generation << '\n'
              << "white noise S.D.: " << noise_sigma << '\n'
              << "------------------------\n";

    CREATE_OUTPUT_DIRECTORY(DEFAULT_OUTPUT_FOLDER)
    initialize_containers(network_size, now_step,
        seed_to_random_number_generation, noise_sigma,
        random_generator, norm_dist, DEFAULT_OUTPUT_FILENAME_NOISE_SERIES,
        stochastic_timeseries_buffer, ofs_stochastic_timeseries
    );


    /* Main calculation loop */
    while (now_step < total_step)
    {
        ++now_step;

        for (int node = 0; node < network_size; ++node) {
            stochastic_timeseries_buffer.push_back(norm_dist(random_generator) * sqrt_dt);
        }

        // Flush to output file and clear buffer
        if (stochastic_timeseries_buffer.size() >= constants::TIMESERIES_BUFFSIZE_THRESHOLD) {
            ofs_stochastic_timeseries.write(
                reinterpret_cast<char*>(&stochastic_timeseries_buffer[0]),
                stochastic_timeseries_buffer.size()*sizeof(float)
            );
            stochastic_timeseries_buffer.clear();
        }
    }

    ofs_stochastic_timeseries.write(
        reinterpret_cast<char*>(&stochastic_timeseries_buffer[0]),
        stochastic_timeseries_buffer.size()*sizeof(float)
    );
    if (ofs_stochastic_timeseries.is_open()) {
        ofs_stochastic_timeseries.close();
    }
    std::cout << "OKAY, stochastic noise time series exported" << std::endl;

    return EXIT_SUCCESS;
}
/**
 * @file IzhikevichModel.cpp
 * @author likchun@outlook.com
 * @brief simulate the dynamics of a network of spiking neurons with Izhikevich's model
 * @version 1.1.2(5)
 * @date 2022-06-20
 * 
 * @copyright free to use
 * 
 * @note to be compiled in C++ version 11 or later with boost library 1.78.0
 * @bug 
 * 
 */

/**
 * change log
 * ver 1.0 - init
 * ver 1.1 - IO update: exported files in /output
 *         - Output update: able to export also recovery variable and current time series
 *         - Input update: synaptic weights matrix nonzero format add network size at line 1
 *         - Add some comments
 * 
 */

/**
 * Optimizations used:
 * 1. spike history truncation
 * 2. auto-determined truncation steps
 * 3. exponential values look-up table
 * 4. loop through only non-zero links
 * 5. reverse loop + break;
 * 6. time series exported in binary format
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
#define DEFAULT_INPUT_FILENAME_PARAMETERS       "vars.txt"
#define DEFAULT_OUTPUT_FOLDER                   "output"

#ifdef _WIN32
#define DEFAULT_OUTPUT_FILENAME_INFO            "output\\info.txt"
#define DEFAULT_OUTPUT_FILENAME_CONTINUATION    "output\\cont.dat"
#define DEFAULT_OUTPUT_FILENAME_SPIKE_TIMESTEP  "output\\spks.txt"
#define DEFAULT_OUTPUT_FILENAME_SPIKE_TIME      "output\\spkt.txt"
#define DEFAULT_OUTPUT_FILENAME_VOLTAGE_SERIES  "output\\memp.bin"
#define DEFAULT_OUTPUT_FILENAME_RECOVER_SERIES  "output\\recv.bin"
#define DEFAULT_OUTPUT_FILENAME_CURRENT_SERIES  "output\\curr.bin"
#define CREATE_OUTPUT_DIRECTORY(__DIR)          if (CreateDirectoryA(__DIR, NULL) || ERROR_ALREADY_EXISTS == GetLastError()) {}\
                                                else { error_handler::throw_error("dir_create", __DIR); }
#else
struct  stat st = {0};
#define DEFAULT_OUTPUT_FILENAME_INFO            "output/info.txt"
#define DEFAULT_OUTPUT_FILENAME_CONTINUATION    "output/cont.dat"
#define DEFAULT_OUTPUT_FILENAME_SPIKE_TIMESTEP  "output/spks.txt"
#define DEFAULT_OUTPUT_FILENAME_SPIKE_TIME      "output/spkt.txt"
#define DEFAULT_OUTPUT_FILENAME_VOLTAGE_SERIES  "output/memp.bin"
#define DEFAULT_OUTPUT_FILENAME_RECOVER_SERIES  "output/recv.bin"
#define DEFAULT_OUTPUT_FILENAME_CURRENT_SERIES  "output/curr.bin"
#define CREATE_OUTPUT_DIRECTORY(__DIR)          if (stat(__DIR, &st) == -1) { mkdir(__DIR, 0700); }
#endif

std::string code_ver = "Version 1.1.2\nBuild 5\nLast Update 20 June 2022";


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
    namespace Izhikevich
    {	
        namespace inh
        {
            constexpr const double a = 0.1;
            constexpr const double b = 0.2;
            constexpr const double c = -65.0;
            constexpr const double d = 2.0;
            constexpr const double threshold_potential = -80.0;
            constexpr const double tau = 6.0;
        };
        namespace exc
        {
            constexpr const double a = 0.02;
            constexpr const double b = 0.2;
            constexpr const double c = -65.0;
            constexpr const double d = 8.0;
            constexpr const double threshold_potential = 0.0;
            constexpr const double tau = 5.0;
        };
        constexpr const double initial_membrane_potential = -65.0;
        constexpr const double initial_recovery_variable = 8;
    }; // Izhikevich
    constexpr const unsigned int TIMESERIES_BUFFSIZE_THRESHOLD = 150000000;
}; // constants


/* Get all parameters and settings from "vars.txt" */
class Parameters
{
private:

    std::vector<std::string> input_param;

public:

    const std::string input_file_synaptic_weights;
    const std::string synaptic_weights_file_input_format;
    const char        synaptic_weights_file_delimiter;
    const double      synaptic_weights_multiplying_factor;

    const std::string output_file_info;
    const std::string output_file_continuation;
    const std::string output_file_spike_timestep;
    const std::string output_file_spike_time;
    const std::string output_file_voltage_time_series;
    const std::string output_file_recover_time_series;
    const std::string output_file_current_time_series;
    const bool        exportVoltageTimeSeries;
    const bool        exportRecoverTimeSeries;
    const bool        exportCurrentTimeSeries;

    const double simulation_time;
    const double delta_time;
    const double noise_sigma;
    const double seed_to_random_number_generation;
    const double driving_current;

    const double suppression_level;
    const int    suppressed_link_type; // +1: suppress exc links, -1: suppress inh links
    const std::vector<int> incoming_link_suppressed_nodes;

    Parameters(std::string filename) :
        input_param(get_input_parameters(filename)),

        input_file_synaptic_weights(input_param[0]), // input_file_synaptic_weights
        synaptic_weights_file_input_format(input_param[1]), // synaptic_weights_file_input_format
        synaptic_weights_file_delimiter(input_param[2] == "tab" ? '\t' : "space" ? ' ' : input_param[3].c_str()[0]), // synaptic_weights_file_delimiter
        synaptic_weights_multiplying_factor(stod(input_param[3])),

        output_file_info(DEFAULT_OUTPUT_FILENAME_INFO),
        output_file_continuation(DEFAULT_OUTPUT_FILENAME_CONTINUATION),
        output_file_spike_timestep(DEFAULT_OUTPUT_FILENAME_SPIKE_TIMESTEP),
        output_file_spike_time(DEFAULT_OUTPUT_FILENAME_SPIKE_TIME),
        output_file_voltage_time_series(DEFAULT_OUTPUT_FILENAME_VOLTAGE_SERIES),
        output_file_recover_time_series(DEFAULT_OUTPUT_FILENAME_RECOVER_SERIES),
        output_file_current_time_series(DEFAULT_OUTPUT_FILENAME_CURRENT_SERIES),
        exportVoltageTimeSeries((input_param[4] == "true") ? true : false), // exportVoltageTimeSeries
        exportRecoverTimeSeries((input_param[6] == "true") ? true : false), // exportRecoverTimeSeries
        exportCurrentTimeSeries((input_param[5] == "true") ? true : false), // exportCurrentTimeSeries

        simulation_time(stod(input_param[7])), // simulation_time
        delta_time(stod(input_param[8])), // delta_time
        noise_sigma(stod(input_param[9])), // noise_sigma
        seed_to_random_number_generation(stod(input_param[10])), // seed_to_random_number_generation
        driving_current(stod(input_param[11])), // strength_of_driving_current

        suppression_level(stod(input_param[12])), // suppression_level
        suppressed_link_type(input_param[13] == "inh" ? -1 : "exc" ? 1 : 0), // suppressed_link_type
        incoming_link_suppressed_nodes(tools::string_to_vector_index(input_param[14], ',')) // incoming_link_suppressed_nodes

    { std::cout << "OKAY, parameters imported from \"" << filename << "\"\n"; }

private:

    std::vector<std::string> get_input_parameters(std::string filename)
    {
        std::vector<std::string> _input_param;
        std::string line, value;
        std::ifstream ifs(filename);
        if (ifs.is_open()) {
            while (std::getline(ifs, line, '\n')) {
                if (line.find('=') != std::string::npos) {
                    std::stringstream ss(line);
                    std::getline(ss, value, '=');
                    std::getline(ss, value, '=');
                    _input_param.push_back(tools::remove_whitespace(value));
                }
            }
            ifs.close();
        } else {
            error_handler::throw_error("file_access", filename);
        }
        return _input_param;
    }
};


/* Neural Network */

/* Import all synaptic weights of a network from a text file (.txt).
   There are two formats this program can read:
   1. "nonzero"
      The first line stores the network size / number of neurons.
      Each remaining line stores the a nonzero link:
      {j i w}, where "j" is the outgoing neuron, "i" is the incoming neuron,
      and "w" is the synaptic weight / coupling strength from neuron j to i.
      The delimiter can be specified.
   2. "full"
      The file stores all synaptic weights in N rows and N columns, where N
      is the network size / number of neurons. It's just the basic matrix
      representation we use daily. */
void import_synaptic_weights(
    const Parameters &par,
    int &network_size,
    std::vector<std::vector<double>> &synaptic_weights
)
{
    std::ifstream ifs;
    ifs.open(par.input_file_synaptic_weights, std::ios::in);
    if (ifs.is_open()) {
        std::vector<std::vector<double>> _synaptic_weights;
        std::vector<double> row_buf;
        std::string line, elem;
        if (par.synaptic_weights_file_input_format == "nonzero") {
            std::getline(ifs, line, '\n');
            network_size = stoi(line);
            synaptic_weights = std::vector<std::vector<double>>(
                network_size, std::vector<double>(network_size, 0));
            while(std::getline(ifs, line, '\n')) {
                std::stringstream ss(line);
                while(std::getline(ss, elem, par.synaptic_weights_file_delimiter)) {
                    if (elem != "") {
                        row_buf.push_back(std::stof(tools::remove_whitespace(elem)));
                    }
                }
                _synaptic_weights.push_back(row_buf);
                row_buf.clear();
            }
            for (size_t i = 0; i < _synaptic_weights.size(); ++i) {
                synaptic_weights[static_cast<int>(_synaptic_weights[i][1])-1][static_cast<int>(_synaptic_weights[i][0])-1] = _synaptic_weights[i][2];
            }
        } else if (par.synaptic_weights_file_input_format == "full") {
            while(std::getline(ifs, line, '\n')) {
                std::stringstream ss(line);
                while(std::getline(ss, elem, par.synaptic_weights_file_delimiter)) {
                    row_buf.push_back(stof(elem)); }
                _synaptic_weights.push_back(row_buf);
                row_buf.clear();
            }
            synaptic_weights = _synaptic_weights;
            network_size = synaptic_weights.size();
        }
        ifs.close();
        for (int i = 0; i < network_size; i++) {
            for (int j = 0; j < network_size; j++) {
                synaptic_weights[i][j] *= par.synaptic_weights_multiplying_factor;
            }
        }
    } else {
        error_handler::throw_error("file_access", par.input_file_synaptic_weights);
    }
}

/* Classify each neuron into INHibitory / EXCitatory / UNCLassified,
   and store the result in "neuron_type". The i-th element of
   the std::vector<int> "neuron_type" stores the type of the (i+1)th
   neuron, which can be -1 (INH) / +1 (EXC) / 0 (UNCL). */
void classify_neuron_type(
    const int network_size,
    std::vector<std::vector<double>> &synaptic_weights,
    std::vector<int> &neuron_type
)
{
    neuron_type = std::vector<int>(network_size);
    for (size_t i = 0; i < synaptic_weights.size(); i++) {
        for (size_t j = 0; j < synaptic_weights.size(); j++) {
            if (synaptic_weights[i][j] > 0) {
                if (neuron_type[j] == -1) {
                    error_handler::throw_error(
                        "neuron_type", std::vector<int>({(int)(i+1), (int)(j+1)})
                    );
                }
                neuron_type[j] = 1;
            }
            else if (synaptic_weights[i][j] < 0) {
                if (neuron_type[j] == 1) {
                    error_handler::throw_error(
                        "neuron_type", std::vector<int>({(int)(i+1), (int)(j+1)})
                    );
                }
                neuron_type[j] = -1;
            }
        }
    }
}

/* Create reference "inhibitory(excitatory)_links_index" which stores
   N lists of indices. The i-th list of indices is all the indices of
   inhibitory (excitatory) presynaptic neuron directing into the
   (i+1)th neuron. Useful for speeding up the calculation. */
void create_quick_link_index_reference(
    std::vector<std::vector<double>> &synaptic_weights,
    std::vector<std::vector<int>> &inhibitory_links_index,
    std::vector<std::vector<int>> &excitatory_links_index
)
{
    std::vector<int> _inh_temp, _exc_temp;
    for (size_t i = 0; i < synaptic_weights.size(); i++) {
        for (size_t j = 0; j < synaptic_weights.size(); j++) {
            if (synaptic_weights[i][j] < 0) { _inh_temp.push_back(j); }
            else if (synaptic_weights[i][j] > 0) { _exc_temp.push_back(j); }
        }
        inhibitory_links_index.push_back(_inh_temp);
        excitatory_links_index.push_back(_exc_temp);
        _inh_temp.clear();
        _exc_temp.clear();
    }
}

const double calculate_standard_deviation_neg(
    std::vector<std::vector<double>> &synaptic_weights
)
{
    int count = 0;
    double sum = 0.0, mean, sd = 0.0;
    std::vector<double> neg_weights;
    for (auto &row : synaptic_weights) {
        for (auto const &elem : row) { if (elem < 0) { sum += elem; ++count; } }
    }
    mean = sum / count;
    for (auto &row : synaptic_weights) {
        for (auto elem : row) { if (elem < 0) { sd += pow(elem - mean, 2); } }
    }
    return sqrt(sd / count);
}

const double calculate_standard_deviation_pos(
    std::vector<std::vector<double>> &synaptic_weights
)
{
    int count = 0;
    double sum = 0.0, mean, sd = 0.0;
    std::vector<double> neg_weights;
    for (auto &row : synaptic_weights) {
        for (auto const &elem : row) { if (elem > 0) { sum += elem; ++count; } }
    }
    mean = sum / count;
    for (auto &row : synaptic_weights) {
        for (auto elem : row) { if (elem > 0) { sd += pow(elem - mean, 2); } }
    }
    return sqrt(sd / count);
}

void suppress_inhibition(
    const Parameters &par,
    std::vector<std::vector<double>> &synaptic_weights
)
{
    const double sd_inh = calculate_standard_deviation_neg(synaptic_weights);

    for (auto &row : synaptic_weights) {
        for (auto &elem : row) {
            if (elem < 0) {
                elem += sd_inh * par.suppression_level;
                if (elem > 0) { elem = 0; }
            }
        }
    }
}

void suppress_inhibition_of_selected(
    const Parameters &par,
    std::vector<std::vector<double>> &synaptic_weights,
    std::vector<std::vector<int>> &inhibitory_links_index
)
// Suppress inhibition incoming links for each neuron in 'incoming_link_suppressed_nodes'
{
    const double sd_inh = calculate_standard_deviation_neg(synaptic_weights);

    for (auto &selected : par.incoming_link_suppressed_nodes) {
        for (auto &link : inhibitory_links_index[selected]) {
            synaptic_weights[selected][link] += sd_inh * par.suppression_level;
            if (synaptic_weights[selected][link] > 0) {
                synaptic_weights[selected][link] = 0; }
        }
    }
}

void suppress_excitation(
    const Parameters &par,
    std::vector<std::vector<double>> &synaptic_weights
)
{
    const double sd_exc = calculate_standard_deviation_pos(synaptic_weights);

    for (auto &row : synaptic_weights) {
        for (auto &elem : row) {
            if (elem > 0) {
                elem -= sd_exc * par.suppression_level;
                if (elem < 0) { elem = 0; }
            }
        }
    }
}

void suppress_excitation_of_selected(
    const Parameters &par,
    std::vector<std::vector<double>> &synaptic_weights,
    std::vector<std::vector<int>> &excitatory_links_index
)
// Suppress excitatory incoming links for each neuron in 'incoming_link_suppressed_nodes'
{
    const double sd_exc = calculate_standard_deviation_pos(synaptic_weights);

    for (auto &selected : par.incoming_link_suppressed_nodes) {
        for (auto &link : excitatory_links_index[selected]) {
            synaptic_weights[selected][link] -= sd_exc * par.suppression_level;
            if (synaptic_weights[selected][link] < 0) {
                synaptic_weights[selected][link] = 0; }
        }
    }
}

void suppress_synaptic_weights(
    const Parameters &par,
    std::vector<std::vector<double>> &synaptic_weights,
    std::vector<std::vector<int>> &inhibitory_links_index,
    std::vector<std::vector<int>> &excitatory_links_index
)
{
    if (par.suppression_level != 0)
    {
        if (par.suppressed_link_type == -1) {
            if (par.incoming_link_suppressed_nodes.empty()) {
                suppress_inhibition(par, synaptic_weights);
            } else {
                suppress_inhibition_of_selected(
                    par, synaptic_weights, inhibitory_links_index);
            }
        } else if (par.suppressed_link_type == 1) {
            if (par.incoming_link_suppressed_nodes.empty()) {
                suppress_excitation(par, synaptic_weights);
            } else {
                suppress_excitation_of_selected(
                    par, synaptic_weights, excitatory_links_index);
            }
        } else {
            error_handler::throw_warning(
                "param_value", "suppressed_link_type", par.suppressed_link_type
            );
        }
    }
}


/* Auxiliary Functions */

void display_info(const Parameters &par, const int network_size)
{
    std::cout << "---------------------------------------------\n"
              << "|network filename:    " << par.input_file_synaptic_weights << '\n'
              << "|network size (N):    " << network_size << '\n'
              << "|simulation time (T): " << par.simulation_time << '\n'
              << "|delta time (dt):     " << par.delta_time << '\n'
              << "|noise generation:\n"
              << "|>strength (sigma):   " << par.noise_sigma << '\n'
              << "|>random number seed: " << par.seed_to_random_number_generation << '\n';
    if (par.driving_current != 0) {
        std::cout << "|driving current:     " << par.driving_current << '\n';
    }
    if (par.suppression_level != 0 || par.synaptic_weights_multiplying_factor != 1) {
        std::cout << "|synaptic weight changes:\n";
        if (par.synaptic_weights_multiplying_factor != 0) {
            std::cout << "|>multiplying factor: " << par.synaptic_weights_multiplying_factor << '\n';
        }
        if (par.synaptic_weights_multiplying_factor != 0) {
            std::cout << "|>suppression lv (k): " << par.suppression_level << '\n'
                      << "|>suppression type:   " << (par.suppressed_link_type == -1 ? "inh" : (par.suppressed_link_type == 1 ? "exc" : "unknown")) << '\n';
        }
    }
    std::cout << "---------------------------------------------" << std::endl;
}

void display_current_datetime()
{
    char datetime_buf[64];
    time_t datetime = time(NULL);
    struct tm *tm = localtime(&datetime);
    strftime(datetime_buf, sizeof(datetime_buf), "%c", tm);
    std::cout << datetime_buf << '\n';
}

/* Estimate the truncation steps needed for the calculation. It uses the
   precision of floating point number / double as a reference to determine
   the threshold of truncation steps that gives accurate results. */
void estimate_truncation_step(
    const Parameters &par,
    std::vector<std::vector<double>> &synaptic_weights,
    int &truncation_step_inh,
    int &truncation_step_exc
)
{
    double w_inh = 0, w_inh_max = 0, w_exc = 0, w_exc_max = 0;
    for (size_t i = 0; i < synaptic_weights.size(); i++) {
        for (size_t j = 0; j < synaptic_weights[i].size(); j++) {
            if (synaptic_weights[i][j] < 0) { w_inh += synaptic_weights[i][j]; }
            else if (synaptic_weights[i][j] > 0) { w_exc += synaptic_weights[i][j]; }
        }
        if (abs(w_inh) > w_inh_max) { w_inh_max = abs(w_inh); }
        if (abs(w_exc) > w_exc_max) { w_exc_max = abs(w_exc); }
    }
    truncation_step_inh = constants::Izhikevich::inh::tau * log(
        datatype_precision::PRECISION_FLOAT * par.synaptic_weights_multiplying_factor * w_inh_max
    ) / par.delta_time;
    truncation_step_exc = constants::Izhikevich::exc::tau * log(
        datatype_precision::PRECISION_FLOAT * par.synaptic_weights_multiplying_factor * w_exc_max
    ) / par.delta_time;
    if (w_inh_max == 0 || par.synaptic_weights_multiplying_factor == 0) { truncation_step_inh = 0; }
    if (w_exc_max == 0 || par.synaptic_weights_multiplying_factor == 0) { truncation_step_exc = 0; }
}

/* Create a look-up table for the exponential spike decay factors.
   Avoid calculating the expensive and redundant exponential function
   multiple times. */
void setup_exp_lookup_table(
    const Parameters &par,
    std::vector<double> &spike_decay_factor_inh,
    std::vector<double> &spike_decay_factor_exc,
    int &truncation_step_inh, int &truncation_step_exc
)
{
    spike_decay_factor_inh = std::vector<double>(truncation_step_inh+1);
    for (int i = 0; i < static_cast<int>(spike_decay_factor_inh.size()); ++i) {
        spike_decay_factor_inh[i] = exp(-i * par.delta_time / constants::Izhikevich::inh::tau);
    }
    spike_decay_factor_exc = std::vector<double>(truncation_step_exc+1);
    for (int i = 0; i < static_cast<int>(spike_decay_factor_exc.size()); ++i) {
        spike_decay_factor_exc[i] = exp(-i * par.delta_time / constants::Izhikevich::exc::tau);
    }
}

void initialize_containers(
    const Parameters &par,
    const int network_size,
    int &now_step,
    int &total_step,
    double &sqrt_dt,
    std::vector<double> &membrane_potential,
    std::vector<double> &recovery_variable,
    std::vector<double> &synaptic_current,
    std::vector<std::vector<int>> &spike_timesteps,
    std::vector<float> &voltage_time_series_buffer,
    std::vector<float> &recover_time_series_buffer,
    std::vector<float> &current_time_series_buffer,
    std::ofstream &ofs_voltage_timeseries,
    std::ofstream &ofs_recover_timeseries, 
    std::ofstream &ofs_current_timeseries,
    boost::random::mt19937 &random_generator,
    boost::random::normal_distribution<double> &norm_dist
)
{
    now_step = 0;
    total_step = static_cast<int>(par.simulation_time/par.delta_time);
    sqrt_dt = sqrt(par.delta_time);
    membrane_potential = std::vector<double>(network_size);
    recovery_variable = std::vector<double>(network_size);
    synaptic_current = std::vector<double>(network_size);
    spike_timesteps = std::vector<std::vector<int>>(network_size);
    fill(membrane_potential.begin(), membrane_potential.end(), constants::Izhikevich::initial_membrane_potential);
    fill(recovery_variable.begin(), recovery_variable.end(), constants::Izhikevich::initial_recovery_variable);
    if (par.exportVoltageTimeSeries) {
        ofs_voltage_timeseries.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        try {
            ofs_voltage_timeseries.open(par.output_file_voltage_time_series, std::ios::trunc | std::ios::binary);
            ofs_voltage_timeseries.close();
            ofs_voltage_timeseries.open(par.output_file_voltage_time_series, std::ios::app | std::ios::binary);
        } catch(std::ifstream::failure const&) {
            error_handler::throw_error("file_access", par.output_file_voltage_time_series);
        }
        voltage_time_series_buffer.reserve(
            static_cast<unsigned int>(
                (constants::TIMESERIES_BUFFSIZE_THRESHOLD / network_size + 1) * network_size
            )
        );
        for (int i = 0; i < network_size; ++i) {
            voltage_time_series_buffer.push_back(membrane_potential[i]);
        }
    }
    if (par.exportRecoverTimeSeries) {
        ofs_recover_timeseries.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        try {
            ofs_recover_timeseries.open(par.output_file_recover_time_series, std::ios::trunc | std::ios::binary);
            ofs_recover_timeseries.close();
            ofs_recover_timeseries.open(par.output_file_recover_time_series, std::ios::app | std::ios::binary);
        } catch(std::ifstream::failure const&) {
            error_handler::throw_error("file_access", par.output_file_recover_time_series);
        }
        recover_time_series_buffer.reserve(
            static_cast<unsigned int>(
                (constants::TIMESERIES_BUFFSIZE_THRESHOLD / network_size + 1) * network_size
            )
        );
        for (int i = 0; i < network_size; ++i) {
            recover_time_series_buffer.push_back(recovery_variable[i]);
        }
    }
    if (par.exportCurrentTimeSeries) {
        ofs_current_timeseries.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        try {
            ofs_current_timeseries.open(par.output_file_current_time_series, std::ios::trunc | std::ios::binary);
            ofs_current_timeseries.close();
            ofs_current_timeseries.open(par.output_file_current_time_series, std::ios::app | std::ios::binary);
        } catch(std::ifstream::failure const&) {
            error_handler::throw_error("file_access", par.output_file_current_time_series);
        }
        current_time_series_buffer.reserve(
            static_cast<unsigned int>(
                (constants::TIMESERIES_BUFFSIZE_THRESHOLD / network_size + 1) * network_size
            )
        );
        for (int i = 0; i < network_size; ++i) {
            current_time_series_buffer.push_back(synaptic_current[i]);
        }
    }
    boost::random::mt19937 _random_generator(par.seed_to_random_number_generation);
    random_generator = _random_generator;
    boost::random::normal_distribution<double> _norm_dist(0, par.noise_sigma);
    norm_dist = _norm_dist;
}

void export_file_info(
    const Parameters &par,
    const int network_size,
    int continuation,
    float time_elapsed,
    int truncation_step_inh,
    int truncation_step_exc
)
{
    char datetime_buf[64];
    time_t datetime = time(NULL);
    struct tm *tm = localtime(&datetime);
    strftime(datetime_buf, sizeof(datetime_buf), "%c", tm);

    if (continuation == -1) {
        std::ofstream ofs;
        ofs.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        try {
            ofs.open(par.output_file_info, std::ios::trunc);
            ofs << code_ver << '\n'
                << "--------------------------------------------------\n"
                << "computation finished at: " << datetime_buf << '\n'
                << "time elapsed: " << time_elapsed << " s\n\n"
                << "[Numerical Settings]" << '\n'
                << "network file name:\t\t\t" << par.input_file_synaptic_weights << '\n'
                << "number of neurons (N):\t\t" << network_size << '\n'
                << "time step size (dt):\t\t" << par.delta_time << '\n'
                << "simulation duration (T):\t" << par.simulation_time << '\n'
                << "random seed number:\t\t\t" << par.seed_to_random_number_generation << '\n'
                << "white noise S.D. (sigma):\t" << par.noise_sigma << '\n'
                << "constant driving current:\t" << par.driving_current << '\n'
                << "spike truncation time:\t\t" << truncation_step_inh*par.delta_time << " (inh)\n"
                << "spike truncation time:\t\t" << truncation_step_exc*par.delta_time << " (exc)\n\n"
                << "[Change in Synaptic Weights]\n"
                << "matrix multiplying factor:\t" << par.synaptic_weights_multiplying_factor << '\n'
                << "suppression level:\t\t\t" << par.suppression_level << '\n';
            std::string type_of_links = par.suppressed_link_type == 1 ? "exc" : "inh";
            ofs << "type of links:\t\t\t\t" << type_of_links << '\n';
            if (par.incoming_link_suppressed_nodes.empty() == false) {
                ofs << "suppressed nodes:\t\t\t" << par.incoming_link_suppressed_nodes[0];
                for (size_t i = 1; i < par.incoming_link_suppressed_nodes.size(); i++) {
                    ofs << ", " << par.incoming_link_suppressed_nodes[i];
                }
                ofs << '\n';
            }
            ofs << "\n[Initial Values]\n"
                << "membrane potential:\t\t\t" << constants::Izhikevich::initial_membrane_potential << '\n'
                << "recovery variable:\t\t\t" << constants::Izhikevich::initial_recovery_variable << '\n' << std::endl;
            ofs.close();
        } catch(std::ifstream::failure const&) {
            error_handler::throw_error("file_access", par.output_file_info);
        }
    } else {
        std::ofstream ofs;
        ofs.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        try {
            ofs.open(par.output_file_info, std::ios::app);
            ofs << "--------------------------------------------------\n"
                << "computation finished at: " << datetime_buf
                << "time elapsed: " << time_elapsed << "\n\n"
                << "extend duration (T) to:\t\t" << par.simulation_time << "\n\n";
            ofs.close();
        } catch(std::ifstream::failure const&) {
            error_handler::throw_error("file_access", par.output_file_info);
        }
    }
}

/* Save the numerical data of the last step, which is useful for
   continuing the numerical simulation. This function will be
   implemented in the future. */
void export_file_continuation(
    const Parameters &par,
    const int network_size,
    int continuation,
    int truncation_step_inh,
    int truncation_step_exc,
    std::vector<double> &membrane_potential_timeseries,
    std::vector<double> &recovery_variable_timeseries,
    std::vector<double> &synaptic_current_timeseries
)
{
    std::ofstream ofs;
    ofs.precision(datatype_precision::DIGIT_DOUBLE);
    ofs.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    try {
        ofs.open(par.output_file_continuation, std::ios::trunc);
        ofs << ++continuation << '|' << par.exportVoltageTimeSeries << '|'
            << par.exportRecoverTimeSeries << '|' << par.exportCurrentTimeSeries << '\n';
        ofs << network_size << '|'
            << par.delta_time << '|' << par.simulation_time << '|'
            << par.seed_to_random_number_generation << '|'
            << par.noise_sigma << '|'
            << truncation_step_inh << '|' << truncation_step_exc << '|'
            << par.synaptic_weights_multiplying_factor << '|'
            << par.suppression_level << '|' << par.suppressed_link_type << '|';
        if (par.incoming_link_suppressed_nodes.empty() == false) {
            ofs << par.incoming_link_suppressed_nodes[0];
            for (size_t i = 1; i < par.incoming_link_suppressed_nodes.size(); i++) {
                ofs << ',' << par.incoming_link_suppressed_nodes[i];
            }
        }
        ofs << '|';
        ofs << constants::Izhikevich::initial_membrane_potential << '|'
            << constants::Izhikevich::initial_recovery_variable << '|'
            << constants::Izhikevich::inh::a << '|' << constants::Izhikevich::inh::b << '|'
            << constants::Izhikevich::inh::c << '|' << constants::Izhikevich::inh::d << '|'
            << constants::Izhikevich::exc::a << '|' << constants::Izhikevich::exc::b << '|'
            << constants::Izhikevich::exc::c << '|' << constants::Izhikevich::exc::d << '|'
            << constants::Izhikevich::inh::threshold_potential << '|'
            << constants::Izhikevich::exc::threshold_potential << '|'
            << constants::Izhikevich::inh::tau << '|'
            << constants::Izhikevich::exc::tau << '|';
        ofs << par.input_file_synaptic_weights << '|'
            << par.synaptic_weights_file_input_format << '|'
            << par.synaptic_weights_file_delimiter << '|'
            << par.output_file_info << '|'
            << par.output_file_spike_timestep << '|'
            << par.output_file_spike_time << '|'
            << par.output_file_voltage_time_series << std::endl;
        ofs << membrane_potential_timeseries[0];
        for (int i = 1; i < network_size; ++i) {
            ofs << '\t' << membrane_potential_timeseries[i]; }
        ofs << '\n' << recovery_variable_timeseries[0];
        for (int i = 1; i < network_size; ++i) {
            ofs << '\t' << recovery_variable_timeseries[i]; }
        ofs << '\n' << synaptic_current_timeseries[0];
        for (int i = 1; i < network_size; ++i) {
            ofs << '\t' << synaptic_current_timeseries[i]; }
        ofs.close();
    } catch(std::ifstream::failure const&) {
        error_handler::throw_error("file_access", par.output_file_continuation);
    }
}

void export_file_spike_data(
    const Parameters &par,
    std::vector<std::vector<int>> &spike_timesteps,
    char delimiter='\t'
)
{
    std::ofstream ofs;
    ofs.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    try {
        ofs.open(par.output_file_spike_timestep, std::ios::trunc);
        ofs << spike_timesteps[0].size();
        for (auto &elem : spike_timesteps[0]) { ofs << delimiter << elem; }
        for (size_t row = 1; row < spike_timesteps.size(); row++) {
            ofs << '\n' << spike_timesteps[row].size();
            for (auto &elem : spike_timesteps[row]) { ofs << delimiter << elem; }
        }
        ofs.close();
    } catch(std::ifstream::failure const&) {
        error_handler::throw_error("file_access", par.output_file_spike_timestep);
    }
    try {
        ofs.open(par.output_file_spike_time, std::ios::trunc);
        ofs << spike_timesteps[0].size();
        for (auto &elem : spike_timesteps[0]) { ofs << delimiter << static_cast<double>(elem*par.delta_time); }
        for (size_t row = 1; row < spike_timesteps.size(); row++) {
            ofs << '\n' << spike_timesteps[row].size();
            for (auto &elem : spike_timesteps[row]) { ofs << delimiter << static_cast<double>(elem*par.delta_time); }
        }
        ofs.close();
    } catch(std::ifstream::failure const&) {
        error_handler::throw_error("file_access", par.output_file_spike_time);
    }
}


int main(int argc, char **argv)
{
    // int mode = 0;           // 0: overwrite mode | 1: continue mode (not implemented)
    int continuation = -1;  // count the number of times of continuation (not implemented)
    int	now_step, diff_step, total_step;
    int	truncation_step_inh, truncation_step_exc;
    int _network_size;
    double spike_contribution_sum, v_temp, sqrt_dt;
    double conductance_inh, conductance_exc;
    std::vector<int> neuron_type;
    std::vector<float> voltage_time_series_buffer;
    std::vector<float> current_time_series_buffer;
    std::vector<float> recover_time_series_buffer;
    std::vector<double> membrane_potential, recovery_variable, synaptic_current;
    std::vector<double> spike_decay_factor_inh, spike_decay_factor_exc;
    std::vector<std::string> input_param;
    std::vector<std::vector<int>> inhibitory_links_index;
    std::vector<std::vector<int>> excitatory_links_index;
    std::vector<std::vector<int>> spike_timesteps;
    std::vector<std::vector<double>> synaptic_weights;
    std::ofstream ofs_voltage_timeseries;
    std::ofstream ofs_current_timeseries;
    std::ofstream ofs_recover_timeseries;
    boost::random::mt19937 random_generator;
    boost::random::normal_distribution<double> norm_dist;

    std::cout << code_ver << "\n...\n";
    display_current_datetime();
    std::cout << "...\nProgram started\n";
    clock_t beg = clock();

    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "-bypass") {}
    }}

    const Parameters par(NO_MAIN_ARGUMENT ? DEFAULT_INPUT_FILENAME_PARAMETERS : argv[1]);

    import_synaptic_weights(par, _network_size, synaptic_weights);
    suppress_synaptic_weights(par, synaptic_weights,
        inhibitory_links_index, excitatory_links_index);
    create_quick_link_index_reference(synaptic_weights,
        inhibitory_links_index, excitatory_links_index);
    classify_neuron_type(_network_size, synaptic_weights, neuron_type);

    const int    network_size = _network_size;
    const double delta_time = par.delta_time;
    const double driving_current = par.driving_current;
    const bool   exportVoltageTimeSeries = par.exportVoltageTimeSeries;
    const bool   exportRecoverTimeSeries = par.exportRecoverTimeSeries;
    const bool   exportCurrentTimeSeries = par.exportCurrentTimeSeries;

    display_info(par, network_size);
    CREATE_OUTPUT_DIRECTORY(DEFAULT_OUTPUT_FOLDER)
    estimate_truncation_step(par, synaptic_weights,
        truncation_step_inh, truncation_step_exc);
    create_quick_link_index_reference(synaptic_weights,
        inhibitory_links_index, excitatory_links_index);
    setup_exp_lookup_table(par, spike_decay_factor_inh,
        spike_decay_factor_exc, truncation_step_inh, truncation_step_exc);
    initialize_containers(par, network_size, now_step, total_step, sqrt_dt,
        membrane_potential, recovery_variable, synaptic_current,
        spike_timesteps, voltage_time_series_buffer,
        recover_time_series_buffer, current_time_series_buffer,
        ofs_voltage_timeseries, ofs_recover_timeseries,
        ofs_current_timeseries, random_generator, norm_dist);

    std::cout << "Running simulation of Izhikevich model ...\n";
    if (par.exportVoltageTimeSeries) { std::cout << "Export membrane potential time series? YES\n"; }
    if (par.exportRecoverTimeSeries) { std::cout << "Export recovery variable time series?  YES\n"; }
    if (par.exportCurrentTimeSeries) { std::cout << "Export current time series?            YES\n"; }
    std::cout << "...patience...\n";
    clock_t lap = clock();

    /* Main calculation loop */
    while (now_step < total_step)
    {
        ++now_step;

        for (int node = 0; node < network_size; ++node)
        {
            v_temp = membrane_potential[node];  // Make a temporary copy so
                                                // that the other variables
                                                // take the membrane potential
                                                // of the previous step
            membrane_potential[node] += (
                (0.04 * membrane_potential[node] * membrane_potential[node]) + (5 * membrane_potential[node])
                + 140 - recovery_variable[node] + synaptic_current[node]
            ) * delta_time
              + norm_dist(random_generator) * sqrt_dt;

            if (neuron_type[node] == -1) {
                recovery_variable[node] += constants::Izhikevich::inh::a * (
                    constants::Izhikevich::inh::b * v_temp - recovery_variable[node]
                ) * delta_time;
            } else {
                recovery_variable[node] += constants::Izhikevich::exc::a * (
                    constants::Izhikevich::exc::b * v_temp - recovery_variable[node]
                ) * delta_time;
            }

            if (membrane_potential[node] >= 30) {
                if (neuron_type[node] == -1) {
                    membrane_potential[node] = constants::Izhikevich::inh::c;
                    recovery_variable[node] += constants::Izhikevich::inh::d;
                } else {
                    membrane_potential[node] = constants::Izhikevich::exc::c;
                    recovery_variable[node] += constants::Izhikevich::exc::d;
                }
                spike_timesteps[node].push_back(now_step);
            }

            conductance_inh = 0;
            for (auto &in_inh : inhibitory_links_index[node]) {
                spike_contribution_sum = 0;
                for (int spk = spike_timesteps[in_inh].size()-1; spk >= 0; --spk) {
                    diff_step = now_step - spike_timesteps[in_inh][spk];
                    if (diff_step < truncation_step_inh) {
                        spike_contribution_sum += spike_decay_factor_inh[diff_step];
                    } else { break; } // Iterate the loop backwards with a conditional break can terminate the loop early to avoid uncessary operations
                }
                conductance_inh -= (synaptic_weights[node][in_inh] * spike_contribution_sum);
            }

            conductance_exc = 0;
            for (auto &in_exc : excitatory_links_index[node]) {
                spike_contribution_sum = 0;
                for (int spk = spike_timesteps[in_exc].size()-1; spk >= 0; --spk) {
                    diff_step = now_step - spike_timesteps[in_exc][spk];
                    if (diff_step < truncation_step_exc) {
                        spike_contribution_sum += spike_decay_factor_exc[diff_step];
                    } else { break; }
                }
                conductance_exc += (synaptic_weights[node][in_exc] * spike_contribution_sum);
            }

            synaptic_current[node] = (
                conductance_exc * (constants::Izhikevich::exc::threshold_potential - v_temp)
                - conductance_inh * (v_temp - constants::Izhikevich::inh::threshold_potential)
            ) + driving_current;
        }

        /* Here, the membrane potential (and other variables) of all neurons
           in a step will be added to a buffer "voltage_time_series_buffer".
           When the buffer is full (i.e., > TIMESERIES_BUFF), the data will
           be dumped into a binary file and the buffer is cleaned. If the
           amount of data is less than "TIMESERIES_BUFF", or after the
           numerical calculation is finished, there will be some "residue"
           data left in the buffer. Those will be treated later. */
        if (exportVoltageTimeSeries) // for >TIMESERIES_BUFF
        {
            for (auto &v : membrane_potential) { voltage_time_series_buffer.push_back(v); }
            // Flush to output file and clear buffer if size exceed "TIMESERIES_BUFF"
            if (voltage_time_series_buffer.size() >= constants::TIMESERIES_BUFFSIZE_THRESHOLD) {
                ofs_voltage_timeseries.write(
                    reinterpret_cast<char*>(&voltage_time_series_buffer[0]),
                    voltage_time_series_buffer.size()*sizeof(float)
                );
                voltage_time_series_buffer.clear();
        }}
        if (exportRecoverTimeSeries)
        {
            for (auto &v : recovery_variable) { recover_time_series_buffer.push_back(v); }
            if (recover_time_series_buffer.size() >= constants::TIMESERIES_BUFFSIZE_THRESHOLD) {
                ofs_recover_timeseries.write(
                    reinterpret_cast<char*>(&recover_time_series_buffer[0]),
                    recover_time_series_buffer.size()*sizeof(float)
                );
                recover_time_series_buffer.clear();
        }}
        if (exportCurrentTimeSeries)
        {
            for (auto &v : synaptic_current) { current_time_series_buffer.push_back(v); }
            if (current_time_series_buffer.size() >= constants::TIMESERIES_BUFFSIZE_THRESHOLD) {
                ofs_current_timeseries.write(
                    reinterpret_cast<char*>(&current_time_series_buffer[0]),
                    current_time_series_buffer.size()*sizeof(float)
                );
                current_time_series_buffer.clear();
        }}
    }

    std::cout << "Completed. Time elapsed: " << (double)(clock() - lap)/CLOCKS_PER_SEC << " s\n";

    export_file_info(par, network_size, continuation, (double)(clock() - beg)/CLOCKS_PER_SEC,
        truncation_step_inh, truncation_step_exc);
    std::cout << "OKAY, simulation info exported" << '\n';
    export_file_continuation(par, network_size, continuation,
        truncation_step_inh, truncation_step_exc,
        membrane_potential, recovery_variable, synaptic_current);
    std::cout << "OKAY, continuation file exported" << '\n';
    export_file_spike_data(par, spike_timesteps);
    std::cout << "OKAY, spike data exported" << std::endl;

    /* The "residue" data aforementioned will be dumped to the binary
       file here. */
    if (exportVoltageTimeSeries) // for <TIMESERIES_BUFF and residue
    {
        ofs_voltage_timeseries.write(
            reinterpret_cast<char*>(&voltage_time_series_buffer[0]),
            voltage_time_series_buffer.size()*sizeof(float)
        );
        if (ofs_voltage_timeseries.is_open()) {
            ofs_voltage_timeseries.close();
        }
        std::cout << "OKAY, membrane potential time series exported" << std::endl;
    }
    if (exportRecoverTimeSeries)
    {
        ofs_recover_timeseries.write(
            reinterpret_cast<char*>(&recover_time_series_buffer[0]),
            recover_time_series_buffer.size()*sizeof(float)
        );
        if (ofs_recover_timeseries.is_open()) {
            ofs_recover_timeseries.close();
        }
        std::cout << "OKAY, recovery variable time series exported" << std::endl;
    }
    if (exportCurrentTimeSeries)
    {
        ofs_current_timeseries.write(
            reinterpret_cast<char*>(&current_time_series_buffer[0]),
            current_time_series_buffer.size()*sizeof(float)
        );
        if (ofs_current_timeseries.is_open()) {
            ofs_current_timeseries.close();
        }
        std::cout << "OKAY, current time series exported" << std::endl;
    }

    return EXIT_SUCCESS;
}
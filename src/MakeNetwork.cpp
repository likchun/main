#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

typedef std::vector<std::vector<double>> Matrix;

#define DEFAULT_MODIFIED_SUFFIX "_modified.txt"


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

    template<typename T>
    void throw_error(std::string _err, std::string _e, T _eval)
    {
        std::string msg;
        if (_err == "var_value") {
            msg = "<invalid variable value>: \""+_e
                +"\" is given an invalid value of "
                +_eval;
        } else {
            msg = "<unknown failure>: errf0";
        }
        std::cerr << "\nError" << msg << std::endl;
        exit(1);
    }

    void throw_error(std::string _err, std::string _e)
    {
        std::string msg;
        if (_err == "file_access") {
            msg = "<file not found>: \""+_e+"\" cannot be found or accessed";
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

    std::pair<double, double> calculate_mean_standard_deviation(
        const char _sign,
        Matrix * synaptic_weights_ptr
    )
    {
        int count = 0;
        double sum = 0.0, mean, sd = 0.0;
        std::vector<double> neg_weights;
        for (auto &row : *synaptic_weights_ptr) {
            if (_sign == -1) {
                for (auto const &elem : row) { if (elem < 0) { sum += elem; ++count; } }
            } else if (_sign == 1) {
                for (auto const &elem : row) { if (elem > 0) { sum += elem; ++count; } }
            } else if (_sign == 0) {
                for (auto const &elem : row) { sum += elem; ++count; }
            } else {
                error_handler::throw_error("var_value", "calculate_standard_deviation()::_sign", _sign);
            }
        }
        mean = sum / count;
        for (auto &row : *synaptic_weights_ptr) {
            if (_sign == -1) {
                for (auto elem : row) { if (elem < 0) { sd += pow(elem - mean, 2); } }
            } else if (_sign == 1) {
                for (auto elem : row) { if (elem > 0) { sd += pow(elem - mean, 2); } }
            } else if (_sign == 0) {
                for (auto const &elem : row) { sum += elem; ++count; }
            } else {
                error_handler::throw_error("var_value", "calculate_standard_deviation()::_sign", _sign);
            }
        }
        return std::make_pair(mean, sqrt(sd / count));
    }

    std::string remove_extension_from_filename(std::string _filename_ext)
    {
        size_t dotpos = _filename_ext.find(".");
        if (dotpos == _filename_ext.npos) {
            return _filename_ext;
        } else {
            return _filename_ext.erase(dotpos);
        }
    }
}; // tools


class Parameters
{
private:

    std::vector<std::string> input_param;

public:

    const std::string input_file_synaptic_weights;
    const std::string synaptic_weights_file_input_format;
    const char        synaptic_weights_file_delimiter;
    const int         network_size;

    const double k;
    const double d;
    const int    cmp;
    const double x;

    Parameters(std::string filename) :
        input_param(get_input_parameters(filename)),

        input_file_synaptic_weights(input_param[0]), // input_file_synaptic_weights
        synaptic_weights_file_input_format(input_param[1]), // synaptic_weights_file_input_format
        synaptic_weights_file_delimiter(input_param[2] == "tab" ? '\t' : "space" ? ' ' : input_param[3].c_str()[0]), // synaptic_weights_file_delimiter
        network_size(stoi(input_param[3])), // network_size

        k(stod(input_param[4])),
        d(stod(input_param[5])),
        cmp(
            input_param[6] == "lessthan" ? -1 :
            input_param[6] == "greaterthan" ? 1 :
            0
        ),
        x(stod(input_param[7]))

    { std::cout << "OKAY, changes are imported from \"" << filename << "\"\n"; }

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


void import_synaptic_weights(
    const Parameters &par,
    Matrix &synaptic_weights
)
{
    std::ifstream ifs;
    ifs.open(par.input_file_synaptic_weights, std::ios::in);
    if (ifs.is_open()) {
        std::vector<std::vector<double>> synaptic_weights_temp;
        std::vector<double> row_buf;
        std::string line, elem;
        if (par.synaptic_weights_file_input_format == "nonzero") {
            synaptic_weights = std::vector<std::vector<double>>(
                par.network_size, std::vector<double>(par.network_size, 0));
            while(std::getline(ifs, line, '\n')) {
                std::stringstream ss(line);
                while(std::getline(ss, elem, par.synaptic_weights_file_delimiter)) {
                    if (elem != "") {
                        row_buf.push_back(std::stof(tools::remove_whitespace(elem)));
                    }
                }
                synaptic_weights_temp.push_back(row_buf);
                row_buf.clear();
            }
            for (size_t i = 0; i < synaptic_weights_temp.size(); ++i) {
                synaptic_weights[static_cast<int>(synaptic_weights_temp[i][1])-1][static_cast<int>(synaptic_weights_temp[i][0])-1] = synaptic_weights_temp[i][2];
            }
        } else if (par.synaptic_weights_file_input_format == "full") {
            while(std::getline(ifs, line, '\n')) {
                std::stringstream ss(line);
                while(std::getline(ss, elem, par.synaptic_weights_file_delimiter)) {
                    row_buf.push_back(stof(elem)); }
                synaptic_weights_temp.push_back(row_buf);
                row_buf.clear();
            }
            synaptic_weights = synaptic_weights_temp;
        }
        ifs.close();
    } else {
        error_handler::throw_error("file_access", par.input_file_synaptic_weights);
    }
}

void export_synaptic_weights(
    std::string filename,
    Matrix &synaptic_weights,
    const char delimiter=' ',
    const bool simplified_format=true
)
{
    std::ofstream ofs;
    ofs.open(filename, std::ios::out);
    if (ofs.is_open()) {
        if (simplified_format) {
            for (size_t j = 0; j < synaptic_weights.size(); ++j) {
                for (size_t i = 0; i < synaptic_weights.size(); ++i) {
                    if (synaptic_weights[i][j] != 0) {
                        ofs << j << delimiter << i << delimiter;
                        ofs << synaptic_weights[i][j] << '\n';
                    }
                }
            }
        } else {
            ofs << synaptic_weights[0][0];
            for (size_t j = 1; j < synaptic_weights.size(); ++j) {
                ofs << delimiter << synaptic_weights[0][j];
            }
            for (size_t i = 1; i < synaptic_weights.size(); ++i) {
                ofs << synaptic_weights[i][0];
                for (size_t j = 1; j < synaptic_weights.size(); ++j) {
                    ofs << delimiter << synaptic_weights[i][j];
                }
            }
        }
        ofs.close();
        std::cout << "DONE. The network is modified and exported as \"" << filename << "\"\n";
    } else {
        error_handler::throw_error("file_access", filename);
    }
}

void adjust_weights(
    const double _mulfactor,
    const double _shiftval,
    const int    _cmp,
    const double _threshold,
    Matrix &synaptic_weights
)
{
    for (size_t i = 0; i < synaptic_weights.size(); ++i) {
        for (size_t j = 0; j < synaptic_weights.size(); ++j) {
            if (_cmp == 1) {
                if (synaptic_weights[i][j] > _threshold) {
                    synaptic_weights[i][j] = _mulfactor * synaptic_weights[i][j] + _shiftval;
                }
            } else if (_cmp == -1) {
                if (synaptic_weights[i][j] < _threshold) {
                    synaptic_weights[i][j] = _mulfactor * synaptic_weights[i][j] + _shiftval;
                }
            }
        }
    }
}


class Enquiry
{
public:

    Enquiry(Matrix * synaptic_weights_ptr) : synaptic_weights_ptr(synaptic_weights_ptr) {}

    std::pair<double, double> get_mean_sd_of_synaptic_weights(std::string _linktype)
    {
        char _lt;
        if (_linktype == "inh") { _lt = -1; }
        else if (_linktype == "exc") { _lt = 1; }
        else if (_linktype == "all") { _lt = 0; }
        else { error_handler::throw_error("var_value", "enquiry()::_linktype", _linktype); }
        return tools::calculate_mean_standard_deviation(_lt, synaptic_weights_ptr);
    }

private:

    Matrix * synaptic_weights_ptr;
};


int main(void)
{
    const Parameters par("change.txt");

    Matrix synaptic_weights;
    import_synaptic_weights(par, synaptic_weights);

    Matrix * synaptic_weights_ptr = &synaptic_weights;
    Enquiry enQ(synaptic_weights_ptr);

    adjust_weights(par.k, par.d, par.cmp, par.x, synaptic_weights);
    // adjust_weights(-1, 0, 1, enQ.get_mean_sd_of_synaptic_weights("inh").second, synaptic_weights);   // Inhibition suppression

    export_synaptic_weights(tools::remove_extension_from_filename(par.input_file_synaptic_weights)+DEFAULT_MODIFIED_SUFFIX, synaptic_weights);

    return EXIT_SUCCESS;
}
/**
 * @file find_covariance_matrices.cpp
 * @author likchun@outlook.com
 * @brief find the equal-time K(0) and time-lagged K(tau) covariance matrices from potential time series binary file generated from start_simulation.
 * Use the command `find_covariance_matrices.exe -h` to look for help.
 * @version 0.2
 * @date 2022-03-14
 * 
 * @note compile with OpenMP [-fopenmp] flag to enable multiprocessing
 * 
 */

#include <filesystem>
#include <numeric>
#include <omp.h>
#include "myinc.h"
#include "getopt.h"

using namespace std;
using namespace myinc;
namespace fs = std::filesystem;
using path = fs::path;

struct Vars
{
    /* Settings */
    int		N		= 4095;
    double	tau		= 0;
    double	dt		= 0.05;
    path	infile	= "memp.dat";
    path	outdir	= "";
	string	logfile	= "";
	bool	gotTau	= false;
	bool	outK0	= false;
	bool	outKT	= false;
} vars;

const int parser(int, char**, char*, Vars&);
const int seek_from_binary(vector<vector<float>>&, path, int, int);

// argv[1]: tau, argv[2]: input path, argv[3]: output directory
int main(int argc, char** argv)
{
	char optstr[] = "hN:t:d:i:o:l:0T";
	if (parser(argc, argv, optstr, vars) == EXIT_FAILURE) { return EXIT_FAILURE; }
	if (!vars.outK0 && !vars.outKT) {
		vars.outK0 = true;
		vars.outKT = true;
	}

	if (vars.outdir != "") {
		fs::create_directory(vars.outdir);
	}

	string log_fname;
	auto [time_fmt1, time_fmt2] = timer.report_time();
	if (vars.logfile.empty()) {
		if (vars.outdir == "") { fs::create_directory("log"); }
		else { fs::create_directory(vars.outdir / "log"); }
		log_fname = "log_"+time_fmt2+".txt";
	}
	else {
		log_fname = "log_"+vars.logfile+".txt";
	}
	outlog.open(log_fname);

	if (!vars.gotTau) {
		outlog << "\nMissingCommandOption: a required command option [-t] is missing, use [-h] flag to seek help\n\n";
		return EXIT_FAILURE;
	}

	outlog << '\n' << time_fmt1 << '\n';
	outlog << "Program started\n\n";
	timer.stopwatch_begin();

	int lag_t = (int)(vars.tau/vars.dt); // lag_t = 20 for tau = 1, dt = 0.05
	path input_path, output_path;
	vector<float> t_avg_v(vars.N), lagL_t_avg_v(vars.N), lagR_t_avg_v(vars.N);
	vector<vector<float>> v(vars.N, vector<float>()), covar_mat(vars.N, vector<float>(vars.N, 0));

	if (vars.infile == "memp.dat") {
		input_path = vars.infile;
	}
	else {
		input_path = vars.infile;
	}
	if (seek_from_binary(v, input_path, 0, vars.N) != 0) { return EXIT_FAILURE; }
	outlog << "Potential time series data read from binary file [" << input_path << "]\n";
	outlog << "Task completed in " << (int)(timer.stopwatch_lap()/1000) << " s\n\n";


	// Calculate equal-time covariance matrix K_ij(0)
	if (vars.outK0)
	{
		#pragma omp parallel for
		for (int i = 0; i < vars.N; i++)
		{
			t_avg_v[i] = average_vect(v[i]);
		}
		#pragma omp parallel for
		for (int i = 0; i < vars.N; i++) {
			for (int j = 0; j < vars.N; j++)
			{
				if (i <= j) {
					covar_mat[i][j] = average_vect(multiply_vect(v[i], v[j])) - t_avg_v[i]*t_avg_v[j];
					covar_mat[j][i] = covar_mat[i][j]; // covariance matrix K(0) is symmatric
				}
			}
		}
		if (vars.outdir == "") {
			output_path = "K_0";
		}
		else {
			output_path = vars.outdir / "K_0";
		}
		if (fwrite_dense_matrix(covar_mat, output_path) == EXIT_SUCCESS) {
			outlog << "Equal-time covariance matrix K(0) written into file [" << output_path << "]" << '\n';
		}
		vector<vector<float>>(vars.N, vector<float>(vars.N, 0)).swap(covar_mat);
		outlog << "Task completed in " << (int)(timer.stopwatch_lap()/1000) << " s\n\n";
	}


	// Calculate time-lagged covariance matrix K_ij(tau)
	if (vars.outKT)
	{
		#pragma omp parallel for
		for (int i = 0; i < vars.N; i++) {
			lagL_t_avg_v[i] = average_vect(v[i], lag_t);
			lagR_t_avg_v[i] = average_vect(v[i], 0, lag_t);
		}
		#pragma omp parallel for
		for (int i = 0; i < vars.N; i++) {
			for (int j = 0; j < vars.N; j++) {
				covar_mat[i][j] = average_vect(multiply_vect(v[i], v[j], lag_t)) - lagL_t_avg_v[i]*lagR_t_avg_v[j];
			}
		}
		if (vars.outdir == "") {
			output_path = "K_tau";
		}
		else {
			output_path = vars.outdir / "K_tau";
		}
		if (fwrite_dense_matrix(covar_mat, output_path) == 0) {
			outlog << "Time-laggeed covariance matrix K(tau) written into file [" << output_path << "]" << '\n';
		}
		vector<vector<float>>(vars.N, vector<float>(vars.N, 0)).swap(covar_mat);
		double task3_time = timer.stopwatch_lap();
		outlog << "Task completed in " << (int)(task3_time/1000) << " s\n\n" << flush;
	}

	return EXIT_SUCCESS;
}

const int parser(int argc, char** argv, char* optstr, Vars& vars)
{
	int c;
	while ((c = getopt(argc, (char**)argv, optstr)) != EOF)
	{
		switch (c)
		{
		case 'h':
			cout << "\nBrief:\n"
				 << "> Find equal-time covariance matrix K(0) and time-lagged\n"
				 << "  covariance matrix K(tau) from potential time series binary\n"
				 << "  file generated from start_simulation.\n\n";
			cout << "Input/Output:\n"
				 << "> in:  [1] potential time series binary data    | memp.dat\n"
				 << "> out: [1] equal-time covariance matrix K(0)    | K_0\n"
				 << "       [2] time-lagged covariance matrix K(tau) | K_tau\n"
				 << "       [3] log file                             | log__no__.txt\n\n";
			cout << "Arguments:\n"
				 << "  -t <flo>\tvalue of tau, required\n"
				 << "  -d <flo>\ttime-step dt, default: 0.05 (ms)\n"
				 << "  -N <int>\tnumber of neurons, default: 4095\n"
				 << "  -i <str>\tpath to input file, default: [memp.dat]\n"
				 << "  -o <str>\tpath to output directory, default: current\n"
				 << "  -l <str>\tlog file name (not for user)\n"
				 << "  -0\t\tcalculate only equal-time K(0)\n"
				 << "  -T\t\tcalculate only time-lagged K(tau)\n"
				 << "  -h\t\tdisplay this help message and exit\n\n";
			return EXIT_FAILURE;
			break;
		case 't':
			vars.tau = stod(optarg);
			vars.gotTau = true;
			break;
		case 'd':
			vars.dt = stod(optarg);
			break;
		case 'N':
			vars.N = stoi(optarg);
			break;
		case 'i':
			vars.infile = optarg;
			break;
		case 'o':
			vars.outdir = optarg;
			break;
		case 'l':
			vars.logfile = optarg;
			break;
		case '0':
			vars.outK0 = true;
			break;
		case 'T':
			vars.outKT = true;
			break;
		case '?':
			cout << "\nInvalidCommandOption: " << argv[optind-1] << " is not a valid option, use [-h] flag to seek help\n\n";
			return EXIT_FAILURE;
			break;
		case ':':
			cout << "\nMissingCommandArgument: " << argv[optind-1] << " takes one argument but none was given, use [-h] flag to seek help\n\n";
			return EXIT_FAILURE;
			break;
		default:
			cout << "\nUnknownError: no handler for option " << (char)c << " or " << c << "\n\n";
			return EXIT_FAILURE;
			break;
		}
	}
	return EXIT_SUCCESS;
}

// vector<vector<float>> v should have number of rows intialized
const int seek_from_binary(vector<vector<float>>& v, path filename, int start_idx, int matrix_size)
{
	ifstream myfile(filename, ios::in | ios::binary);
	if (myfile.is_open()) {
		float content_buf;
		myfile.seekg(start_idx*sizeof(float), ios::beg);
		for (size_t i = 0; i < v.size(); i++) {
			myfile.read(reinterpret_cast<char*>(&content_buf), sizeof(float));
			v[i].push_back(content_buf);
		}
		while (myfile.eof() == 0) {
			myfile.seekg((matrix_size-v.size())*sizeof(float), ios::cur);
			for (size_t i = 0; i < v.size(); i++) {
				myfile.read(reinterpret_cast<char*>(&content_buf), sizeof(float));
				v[i].push_back(content_buf);
			}
		}
		for (size_t i = 0; i < v.size(); i++) {
			v[i].pop_back();
		}
		myfile.close();
	} else {
		outlog << "FileAccessError: potential time series data file [" << filename << "] cannot be accessed\n";
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
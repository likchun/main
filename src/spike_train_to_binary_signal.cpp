/**
 * @file spike_train_to_binary_signal.cpp
 * @author likchun@outlook.com
 * @brief covert spike trains data "spks.txt" to binary digital time series
 * @version 0.1
 * @date 2022-04-02
 * 
 * @copyright free to use
 * 
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

struct Vars
{
	/* Settings */
	int		N		= 4095;
	double	Tn		= 10000;
	double	dt		= 0.05;

	string	infile	= "spks.txt";
	string	outfile	= "spk_bin_signal.dat";
} vars;

int import_spks(vector<vector<int>>&, string&);

int main()
{
	vector<int> spk_counter(vars.N, 0);
	vector<vector<int>> spks;

	if (import_spks(spks, vars.infile) == EXIT_FAILURE) { return EXIT_FAILURE; }
	cout << "Spike trains data read from text file \"" << vars.infile << "\"\n";

	for (int node = 0; node < vars.N; node++) {
		if (spks[node].size() == 0) {
			spks[node].push_back(-1); // so that the later loop runs correctly
		}
	}

	int bin_signal, now_step = 0, total_step = (int)(vars.Tn/vars.dt);
	ofstream ofs(vars.outfile, ios::out | ios::binary);

	cout << "Writing spiking time series\n";
	while (now_step < total_step)
	{
		now_step++;
		for (int node = 0; node < vars.N; node++) {
			if (spks[node][spk_counter[node]] == now_step) {
				bin_signal = 1;
				//write 1 to binary file
				ofs.write(reinterpret_cast<const char *>(&bin_signal), sizeof(bin_signal));
				spk_counter[node]++;
			} else {
				bin_signal = 0;
				//write 0 to binary file
				ofs.write(reinterpret_cast<const char *>(&bin_signal), sizeof(bin_signal));
			}
		}
	}
	ofs.close();
	cout << "Spiking time series written into binary file \"" << vars.outfile << "\"\n";

	return EXIT_SUCCESS;
}

int import_spks(vector<vector<int>> &spks, string &infile_spks)
{
	vector<int> row_buf;
	string line, elem;
	ifstream ifs(infile_spks);
	if (ifs.is_open()) {
		while(getline(ifs, line, '\n')) {
			stringstream ss(line);
			while(getline(ss, elem, '\t')) { row_buf.push_back(stof(elem)); }
			row_buf.erase(row_buf.begin()); // delete the number of spikes
			spks.push_back(row_buf);
			row_buf.clear();
		}
		ifs.close();
	} else {
		cout << "FileAccessError: spike train data file \""
			 << infile_spks << "\" cannot be accessed\n";
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

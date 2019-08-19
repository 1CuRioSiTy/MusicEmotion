function features = data_output_function_py_single_file_cnn(path)
    Fs = 22050;
    DURA = 29;
    music_data = audioread(path);
    
    n_sample = length(music_data);
    n_sample_fit = int64(DURA*Fs);
    if n_sample < n_sample_fit
        music_data = [music_data;zeros(int64(DURA*Fs) - n_sample, 1)];
        
    elseif n_sample > n_sample_fit
        music_data = music_data(int64((n_sample-n_sample_fit)/2):int64((n_sample+n_sample_fit)/2));
    end

	T = 8192;
	N = 639450;% 22050 * 29;
	filt_opt.Q = [8 1];
	filt_opt.J = T_to_J(T, filt_opt);
	scat_opt.M = 2;
	Wop = wavelet_factory_1d(N, filt_opt, scat_opt);
	features = format_scat(log_scat(renorm_scat(scat(music_data, Wop))));
    
% musicpath='D:\dh\DL\music\MTT_wav\0\american_bach_soloists-j_s__bach__cantatas_volume_v-01-gleichwie_der_regen_und_schnee_vom_himmel_fallt_bwv_18_i_sinfonia-0-29.wav'
% data_output_function_py_single_file_cnn(musicpath)


% musicpath='demo/test.wav'
% data_output_function_py_single_file_cnn(musicpath)

	
	



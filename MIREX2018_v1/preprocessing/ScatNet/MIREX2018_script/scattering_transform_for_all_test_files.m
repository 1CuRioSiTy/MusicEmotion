function scattering_transform_for_all_input_files(current_path, input_data_path)
    fid = fopen(input_data_path);
    disp('scattering transform for all test files:')
    tline = fgetl(fid);
    while ischar(tline) && length(tline) > 0

        disp(tline)
        scat_coeffs = data_output_function_py_single_file(tline);

        % write to file
        save_file_name = replace(tline, '/', '_');
        save_foler = [current_path,'/preprocessing','/scat_coefficients_test/'];
        fid2 = fopen([save_foler,save_file_name,'.scat'], 'w');
        for i=1:size(scat_coeffs,1) % 1:433
            fprintf(fid2,'%.11f,',scat_coeffs(i,1:size(scat_coeffs,2)-1));  % 1:113
            fprintf(fid2,'%.11f\n',scat_coeffs(i,size(scat_coeffs,2))); % 114
        end
        
        fclose(fid2);

        % get contents of next line 
        tline = fgetl(fid);
    end
    fclose(fid);

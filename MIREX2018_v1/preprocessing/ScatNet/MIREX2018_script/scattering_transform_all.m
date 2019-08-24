function scattering_transform_all(scratch_path, featureExtractionListFile_path, current_line, record_path)
	try
	    fid = fopen(featureExtractionListFile_path);
	    disp('scattering transform:')
	    tline = fgetl(fid);
	    mkdir([scratch_path,'/preprocessing','/scat_coefficients/']);
	    addpath(genpath(scratch_path));
	    addpath(genpath([scratch_path,'/preprocessing']));
	    addpath(genpath([scratch_path,'/preprocessing','/scat_coefficients/']));
	    line_num = 0;
	    while ischar(tline) && length(tline) > 0
		if line_num >= str2num(current_line)
			X = sprintf('Extracting: file %d --> %s.',line_num,tline);
			disp(X)
			scat_coeffs = data_output_function_py_single_file(tline);
		
			% write to file
			save_file_name = replace(tline, '/', '-');
			save_foler = [scratch_path,'/preprocessing','/scat_coefficients/'];
			save_path = [save_foler,save_file_name,'.scat'];
			fid2 = fopen(save_path, 'w');
			for i=1:size(scat_coeffs,1) % 1:433
			    fprintf(fid2,'%.11f,',scat_coeffs(i,1:size(scat_coeffs,2)-1));
			    fprintf(fid2,'%.11f\n',scat_coeffs(i,size(scat_coeffs,2)));
			end
		
			fclose(fid2);
			
			% record current file line number.
			fid3 = fopen(record_path, 'w');
			fprintf(fid3, '%d', line_num);
			fclose(fid3);
		end

		% get contents of next line 
		tline = fgetl(fid);
		line_num = line_num+1;
	    end
	    fclose(fid);
	catch exception
		disp('Errors occured in matlab scripts.')
		exit;
	end

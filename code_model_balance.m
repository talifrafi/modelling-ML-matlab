clear; clc;

% Import Data
data = readtable('balance-scale.xlsx');

% Data Training
input = data{1:438,1:4}; 
target = data{1:438,5:7}; 
[input_baris,input_kolom] = size(input); 
[target_baris,target_kolom] = size(target); 
banyak_data = input_baris;
input = input; %normalisasi data dengan zscore
               %kalau ga di normalisasi jadiin input = input

% Data Testing
input_test = data{439:625,1:4}; 
target_test = data{439:625,5:7}; 
banyak_test = length(input_test);
input_test = input_test; %normalisasi data dengan zscore
                         %kalau ga di normalisasi jadiin input = input

% Inisialisasi ANN
banyak_input = input_kolom; %jumlah unit input layer
banyak_hidden = 9; %jumlah unit hidden layer
banyak_output= target_kolom; %jumlah unit output layer
alpha = 0.2; %learning rate
miu = 0.5; %koefisien momentum
rng(24,"philox");

% Inisialiasi Bobot: Input layer ke hidden layer (Nguyen-Widrow)
beta = 0.7*banyak_hidden^(1/banyak_input);
v_ij = rand(banyak_input,banyak_hidden) - 0.5;
for i = 1:banyak_hidden
    norma(i) = sqrt(sum(v_ij(:,i).^2));
    v_ij(:,i) = (beta*v_ij(:,i))/norma(i);
end
v_0j = (rand(1,banyak_hidden) - 0.5)*beta;

% Inisialiasi Bobot: Input layer ke hidden layer (Random)
%v_ij = rand(banyak_input,banyak_hidden) - 0.5;
%v_0j = rand(1,banyak_hidden) - 0.5;

% Inisialiasi Bobot: Hidden layer ke output layer (Nguyen-Widrow)
%beta = 0.7*banyak_output^(1/banyak_hidden);
%w_jk = rand(banyak_hidden,banyak_output) - 0.5;
%for i = 1:banyak_output
    %norma(i) = sqrt(sum(w_jk(:,i).^2));
    %w_jk(:,i) = (beta*w_jk(:,i))/norma(i);
%end
%w_0k = (rand(1,banyak_output) - 0.5)*beta;

% Inisialiasi Bobot: Hidden layer ke output layer (Random)
w_jk = rand(banyak_hidden,banyak_output) - 0.5;
w_0k = rand(1,banyak_output) - 0.5;

% Training
banyak_epoch = 1000; 
error_min = 0.01;
flag = 0; 
epoch_iter = 1; 
delta_wjk_old = 0; 
delta_w0k_old = 0; 
delta_vij_old = 0; 
delta_v0j_old = 0;

while flag == 0 && epoch_iter <= banyak_epoch
    train_true = 0; 
    train_false = 0;
    for n=1:banyak_data
      
      % Feedforward 
      xi = input(n,:); 
      ti = target(n,:); 
      
      % Input layer ke hidden layer 
      z_inj = xi*v_ij + v_0j; % hitung sinyal input dengan bobot
      for j=1:banyak_hidden;  
          zj(1,j) = 1/(1+exp(-z_inj(1,j))); % hitung nilai aktivasi (Sigmoid) setiap unit hidden sebagai hasil unit hidden
      end
      
      % Hidden layer ke output layer 
      y_ink = zj*w_jk + w_0k; % hitung sinyal input (hasil hidden zj) dengan bobot
      for k=1:banyak_output
          yk(1,k) = 1/(1+exp(-y_ink(1,k))); % hitung nilai aktivasi (Sigmoid) setiap unit output sebagai hasil jaringan
          if yk(1,k) >= 0.7 %kuantisasi hasil aktivasi
              yk(1,k) = 1; 
          end
          if yk(1,k) <= 0.3
              yk(1,k) = 0; 
          end
      end
      
      % Simpan error 
      error(1,n) = 0.5*sum((yk - ti).^2); %kuadratik
      
      % Kalkulasi recognition rate train
      [value_train, index_train] = max(yk); 
      y_train = zeros(size(ti)); 
      y_train(1,index_train) = 1; 
      if y_train == ti
          train_true = train_true + 1; 
      else
          train_false = train_false + 1;
      end
       
      % Backpropagation
      % Output layer ke hidden layer
      dok = (yk - ti).*(yk).*(1-yk); % hitung sinyal error
      delta_wjk = alpha*zj'*dok + miu*delta_wjk_old; % modifikasi bobot dengan momentum
      delta_w0k = alpha*dok + miu*delta_w0k_old; % modifikasi bias dengan momentum
      delta_wjk_old = delta_wjk;
      delta_w0k_old = delta_w0k;
      
      % Hidden layer ke input layer 
      doinj = dok*w_jk';
      doj = doinj.*zj.*(1-zj); % hitung sinyal error
      delta_vij = alpha*xi'*doj + miu*delta_vij_old; % modifikasi bobot (momentum)
      delta_v0j = alpha*doj + miu*delta_v0j_old; % modifikasi bias (momentum)
      delta_vij_old = delta_vij; 
      delta_v0j_old = delta_v0j;
      
      % Update bobot dan bias (new) dengan momentum
      w_jk = w_jk - delta_wjk; 
      w_0k = w_0k - delta_w0k; 
      v_ij = v_ij - delta_vij; 
      v_0j = v_0j - delta_v0j; 
    end
    epoch_error(1,epoch_iter) = sum(error)/banyak_data;
    
    if epoch_error(1,epoch_iter) < error_min
       flag = 1; 
    end 
    
    epoch_iter = epoch_iter + 1;
end

epoch_iter = epoch_iter - 1;
figure;
plot(epoch_error); 
ylabel('Error per epoch'); 
xlabel('Epoch')

disp("Error per epoch = "+ min(epoch_error) +""); 
disp("Error akhir = "+ epoch_error(1,epoch_iter) +"");

recog_rate_train = (train_true/banyak_data)*100;
disp("Recognition rate train = "+ recog_rate_train +" %");

% Testing
test_true = 0; 
test_false = 0;
for n=1:banyak_test
    
    % Feedforward
    xi_test = input_test(n,:); 
    ti_test = target_test(n,:);
    
    % Input layer ke hidden layer
    z_inputj_test = xi_test*v_ij + v_0j; 
    for j=1:banyak_hidden
        zj_test(1,j) = 1/(1+exp(-z_inputj_test(1,j))); 
    end
    
    % Hidden layer ke output layer 
    y_inputk_test = zj_test*w_jk + w_0k;
    for k=1:banyak_output
      yk_test(1,k) = 1/(1+exp(-y_inputk_test(1,k))); 
      if yk_test(1,k) >= 0.7
          yk_test(1,k) = 1;
      end
      if yk_test(1,k) <= 0.3
          yk_test(1,k) = 0; 
      end
    end
    
    % Simpan error 
    error_test(1,n) = 0.5*sum((yk_test - ti_test).^2); % kuadratik
    
    % Kalkulasi recognition rate test
    [value_test, index_test] = max(yk_test); 
    y_test = zeros(size(ti_test)); 
    y_test(1,index_test) = 1; 
    if y_test == ti_test
        test_true = test_true + 1; 
    else
        test_false = test_false + 1; 
    end
end

avgerrortest = sum(error_test)/banyak_test;
disp("Error average test = "+ avgerrortest +"");

recog_rate_test = (test_true/banyak_test)*100;
disp("Recognition rate test = "+ recog_rate_test +" %");
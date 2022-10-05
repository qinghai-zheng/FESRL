function [U,V,Z,Ws,labels] = Weighted_FEMSRL(X,A,k,lambda,ridx)
%ISRL - Intrinsic Subspace Representation Learning
%
% Syntax: [U,V] = ISRL(X,A,gt,lambda)
%
% Input:
% X: multi-view data, X is defined as cell(1,v), where v is the number of views
%    the format of X{i} is n_dimension*n_samples
% A: anchor of the multi-view data, A is also defined as cell(1,V)
% r: exponent of weights
% lambda: hyper-parameter used in the proposed ISRL
% ridx: index of the weights for different views
%
% Output:
% U: cell(1,v), matrices for diversity information, U{i} is n_anchor_number*n_anchor_number
% V: n_anchor_number*n_number, it is the intrinsic subspace representation used for clustering
% labels: clustering results
%
% Copyright: zhengqinghai@stu.xjtu.edu.cn
% 2021/11/05

v = size(X,1)+size(X,2)-1; % the number of views
n = size(X{1},2); % the number of samples, n_dimension
m = size(A{1},2); % the number of anchor samples, n_anchor_number
% we set Z = U*V, s.t. U'U = I
U = cell(1,v); % m*m
V = ones(n,m)/m; % n*m
Y = cell(1,v);
Z = cell(1,v);
Ws = ones(v,1)/v;
for i = 1:v
    Z{i} = ones(n,m)/m;
    U{i} = ones(m,m)/m;
    Y{i} = zeros(n,m);
end

quadprog_options = optimset( 'Algorithm','interior-point-convex','Display','off');

iter = 0;
Isconverg = 0;
max_iter = 20; % usually achieve convergence in 20 iterations
epson = 1e-7;

rho = 10e-5; max_rho = 10e10; pho_rho = 2;

%% optimization
while (Isconverg == 0)
    % ����Ĵ���һ��Ҫע���ά�ȵĴ�����ͬʱװ�õĲ���һ��Ҫ�ԣ���Ȼ��һ��������ǳ��󣡣���?
    % �������漰��Z��V��U��Y�⼸����һ��Ҫ��������Ƿ���װ�õĲ���������?
    % updating Z
    for i = 1:v
        Z_tmp1 = (Ws(i)^ridx)*A{i}'*A{i}+ 0.5*rho*eye(m);
        Z_tmp1 = 2*Z_tmp1;
        Z_tmp1 = (Z_tmp1'+Z_tmp1)/2;
        
        Z_i = Z{i};
        X_i = X{i};
        A_i = A{i};
        Y_i = Y{i};
        UV_i = U{i}*V';
        Ws_i = Ws(i);
        parfor j = 1:n
            Z_tmp2 = -2*(Ws_i^ridx)*((X_i(:,j))'*A_i)+Y_i(j,:)-rho*(UV_i(:,j))';
            Z_i(j,:) = quadprog(Z_tmp1,Z_tmp2',[],[],ones(1,m),1,zeros(m,1),ones(m,1),Z_i(j,:),quadprog_options);
        end
        Z{i}=Z_i; 
    end

    % update V
    V_tmp1 = (lambda+0.5*rho*v)*eye(m); % V_tmp1 is defined as left form, since U'*U=I
    V_tmp1 = 2*V_tmp1;
    V_tmp1 = (V_tmp1'+V_tmp1)/2;
    for j = 1:n
        V_tmp2 = zeros(1,m);
        for i = 1:v
            V_tmp2 = V_tmp2 + 0.5*Y{i}(j,:)*U{i}+0.5*rho*Z{i}(j,:)*U{i};
        end
        V_tmp2 = -2*V_tmp2;
        V(j,:) = quadprog(V_tmp1,V_tmp2',[],[],ones(1,m),1,zeros(m,1),ones(m,1),V(j,:),quadprog_options);
    end

    % updating U
    for i = 1:v
        U_a = Z{i} + Y{i}/rho;
        U_b = V'*U_a;
        [svd_U,~,svd_V] = svd(U_b,'econ');
        U{i} = svd_V*svd_U';
    end

    % update rho and Y
    for i = 1:v
        tmp = Y{i}' + rho*(Z{i}'-U{i}*V');
        Y{i} = tmp';
    end
    rho = min(rho*pho_rho, max_rho);
    
    % update Ws 
    tmp_X_AZ = zeros(v,1);
    for i = 1:v
        tmp_X_AZ(i) = norm(X{i}-A{i}*Z{i}','fro');
    end
    tmp_X_AZ_F = tmp_X_AZ.^-1;
    tmp_X_AZ_F_tmp = 1/sum(tmp_X_AZ_F);
    Ws = tmp_X_AZ_F_tmp*tmp_X_AZ_F;
    
    iter = iter + 1;
    Isconverg = 1;
    for i = 1:v
        % convergence condiction: norm(Z{i}-V*U{i}',inf) < epson
        if (norm(Z{i}-V*U{i}',inf)>epson)
            Isconverg = 0;
        end
    end

    if (iter>max_iter)
        Isconverg  = 1;
    end
end

Sbar = V;

[Q,~,~] = mySVD(Sbar,k); 

rng(1234,'twister') % set random seed for re-production
labels=litekmeans(Q, k, 'MaxIter', 100,'Replicates',10);%kmeans(U, c, 'emptyaction', 'singleton', 'replicates', 100, 'display', 'off');

end
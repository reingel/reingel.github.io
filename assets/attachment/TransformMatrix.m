classdef TransformMatrix
    % 2x2 Transformation Matrix that can visualize
    
    properties
        A
    end
    
    methods
        function obj = TransformMatrix(A)
            obj.A = A;
        end
        
        function [V,D] = eig(obj)
            [V,D] = eig(obj.A);
        end
        
        function dispTransformMatrix(obj)
            disp('Transform Matrix:');
            disp(num2str(obj.A));
        end
        
        function dispEigenvectors(obj)
            [V,D] = eig(obj);
            
            disp('Eigenvector 1:');
            disp(num2str(V(:,1),'%4.2f'));
            disp('Eigenvector 2:');
            disp(num2str(V(:,2),'%4.2f'));
        end
        
        function dispEigenvalues(obj)
            [V,D] = eig(obj);
            
            disp('Eigenvalues:');
            disp([num2str(D(1,1),'%4.3f') ' ' num2str(D(2,2),'%4.3f')]);
            disp([num2str(D(1,1)) ' ' num2str(D(2,2))]);
        end
        
        function showEigenvalues(obj)
            [V,D] = eig(obj);
            
            figure(200)
            clf
            hold on
            plot(1e9*[-1 1],[0 0],'k:'); % x-axis
            plot([0 0],1e9*[-1 1],'k:'); % y-axis
            t = linspace(0,2*pi);
            plot(cos(t),sin(t),'k:'); % unit circle
            for i = 1:2,
                x = real(D(i,i)); if abs(x) < eps, x = 0; end
                y = imag(D(i,i)); if abs(y) < eps, y = 0; end
                plot(x,y,'xb'); % eigenvalue
                
                if x ~= 0 && y ~= 0,
                    str = [num2str(x) ' + ' num2str(y) 'i'];
                elseif x == 0 && y ~= 0,
                    str = [num2str(y) 'i'];
                elseif x ~= 0 && y == 0,
                    str = num2str(x);
                else
                    str = '0';
                end
                
                text(x + 0.01,y + 0.1*(i*2-3),str);
            end
            hold off
            axis square
        end
        
        function showField(obj,xrange,yrange,iuv,itv,idv,iev,iqv)
            % xrange : x vector of field
            % yrange : y vector of field
            % iuv : boolean to show or not untransformed vectors
            % itv : boolean to show or not transformed vectors
            % idv : boolean to show or not defference vectors
            % iev : boolean to show or not eigenvectors
            % iqv : boolean to show or not quadratic values
            
            [V,D] = eig(obj);
            
            nx = length(xrange);
            ny = length(yrange);
            n = nx * ny;
            [x0,y0] = meshgrid(xrange,yrange); % meshgrid for untransformed vectors
            xo = x0*0; % meshgrid for origin
            yo = y0*0;
            x1 = x0; % meshgrid for transformed vectors
            y1 = y0;
            u = x0; % meshgrid for difference vectors
            v = y0;
            qd = x0; % meshgrid for quadratic values
            
            for i = 1:ny,
                for j = 1:nx,
                    k = (i - 1) * nx + j;
                    v0 = [xrange(j); yrange(i)]; % before transformation
                    v1 = obj.A * v0; % after transformation
                    qv = v0' * v1; % quadratic value
                    
                    x1(i,j) = v1(1);
                    y1(i,j) = v1(2);
                    u(i,j) = v1(1) - v0(1);
                    v(i,j) = v1(2) - v0(2);
                    qd(i,j) = qv;
                end
            end
            
            figure(100)
            clf
            hold on
            plot(1e9*[-1 1],[0 0],'k:'); % x-axis
            plot([0 0],1e9*[-1 1],'k:'); % y-axis
            if iuv,
                quiver(xo,yo,x0,y0,0,'g');
            end
            if itv,
                quiver(xo,yo,x1,y1,0,'b');
            end
            if idv,
                quiver(x0,y0,u,v,0,'r');
            end
            if iev,
                k = 5;
                plot([0 V(1,1)]*D(1,1)*k,[0 V(2,1)]*D(1,1)*k,'k');
                plot([0 V(1,2)]*D(2,2)*k,[0 V(2,2)]*D(2,2)*k,'k');
            end
            
            if iqv,
                for i = 1:ny,
                    for j = 1:nx,
                        text(xrange(j),yrange(i),num2str(qd(i,j),'%4.2f'));
                    end
                end
            end
            axis square
            
        end
    end
    
end


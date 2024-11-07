function knngraph = getKNNGraph(graph,knn)
%input:
%graph is a matrix;
%knn is the parameter of KNN
%output:
%knngraph is the KNN of the graph per row

[m,n] = size(graph);

[srtdDt,srtdIdx] = sort(graph,2,'descend');
dt               = srtdDt(:,1:knn+1);
nidx             = srtdIdx(:,1:knn+1);

% only find the sparse knn graphs
i = repmat(1:m,1,knn+1);
knngraph = sparse(i(:),double(nidx(:)),dt(:),m,n); 

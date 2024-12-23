close all
clear all
% Définir les seuils pour détecter le bleu (ajuster si nécessaire)
hueThresholdLow = 0.55; % Borne basse de la teinte
hueThresholdHigh = 0.75; % Borne haute de la teinte
saturationThreshold = 0.3; % Seuil minimum pour la saturation
valueThreshold = 0.2; % Seuil minimum pour la luminosité

new_p = zeros(8,2);
for i = 1:48
    % Charger l'image
    I = imread(['set3_img (',num2str(i),').jpg']);
    I = imrotate(I, -90);
    
    % Convertir en espace HSV
    hsvImage = rgb2hsv(I);
    
    % Séparer les canaux HSV
    hue = hsvImage(:,:,1);        % Teinte
    saturation = hsvImage(:,:,2); % Saturation
    value = hsvImage(:,:,3);      % Valeur
    
    % Créer un masque binaire pour les pixels bleus
    blueMask = (hue >= hueThresholdLow) & (hue <= hueThresholdHigh) & ...
               (saturation >= saturationThreshold) & ...
               (value >= valueThreshold);
    
    % Optionnel : Nettoyer le masque avec des opérations morphologiques
    blueMask = imopen(blueMask, strel('square', 3)); % Suppression de bruit
    blueMask = imclose(blueMask, strel('square', 3)); % Remplir les trous
    
    % Créer l'image en noir et blanc
    outputImage = uint8(blueMask) * 255; % Convertir le masque en image 8 bits
    
%     imshow(outputImage);
%     title('Carrés bleus détectés (en blanc)');

    % ----------- Matrice H -------
     
    
     % Etiquetage région 4 connexité 
    [Ir,numb]=bwlabel(outputImage); % Ir, etiquette de chaque pixel - numb, nb de région

    % Affichage de l'image labelisée
%     figure;
%     imagesc(Ir); % Affiche la matrice labellisée
%     colormap(jet); % Utilise une carte de couleurs (jet pour des couleurs distinctes)
%     colorbar; % Ajoute une barre de couleurs pour référence
%     title('Carrés bleus détectés');

    s=zeros(numb,1); %stocker la taille 
    c=zeros(numb,2); %pour stocker les coordonnées x,y
    
    for j=1:numb % parcourir tous les carrés
        [y,x]=find(Ir==j); % cherche les coord des pixels d'étiquette j
        c(j,:)=[mean(x),mean(y)]; % centre de gravité du carré bleu
        s(j)=length(x); % nombre de pixels d'étiquette j
    end

    new_p = zeros(8,2);
    p_calcul = c;
    p1 = [0 0; 0 5; 0 10; 5 0; 5 10; 10 0; 10 5; 10 10]*10; % [mm]10
    if i == 1
        %pour la première itération seulement 
        new_p = [1.2923 2.4230; 1.5804 2.6966; 1.9241 3.0191; 1.6492 2.3139; 2.3148 2.8287; 1.9831 2.2144; 2.2960 2.4276; 2.6466 2.6699]*1e3;  
    else 
        for j = 1:8 
            distances = sqrt(sum((p_calcul(j,:) - old_p).^2, 2));
            % Trouver l'indice du point le plus proche
            [~, idx] = min(distances);
            % Extraire le point le plus proche
            new_p(idx,:) = p_calcul(j,:);
        end
    end 
    old_p = new_p;
    p_sorted = new_p
    


    p2 = p_sorted;

    %Extractionn du nombre de points
    n = size(p1,1);

    %Initialisation de la matrice H 
    A = zeros(2*n, 9);
    %Remplissage de la matrice A 
    for j = 1:n
        x = p1(j,1);
        y = p1(j,2);
        xp = p2(j,1);
        yp = p2(j,2);

    %A(2*i-1,:) = [0 0 0 -x -y -1 x*yp y*yp yp];
    %A(2*i,:) = [x y 1 0 0 0 -x*xp -y*xp -xp];

        A(2*j,:) = [0 0 0 x y 1 -x*yp -y*yp -yp];
        A(2*j-1,:) = [x y 1 0 0 0 -x*xp -y*xp -xp];
    
    end

    %Utilisation de la méthode DLT 
    [U,S,V] = svd(A);
    h = V(:,end);
    H = reshape(h,[3 3])';
 
 
    close all;
    figure; % Create a new figure window
    imshow(I); % Display the image
    hold on;

    xpt1=zeros(n,3);
    for j = 1:n
        xpt1(j,:) = H*[p1(j,:) 1]'*21;
        xpt1(j,:) = xpt1(j,:)/xpt1(j,3);
    end

    x = xpt1(:, 1); % Extraire les coordonnées X
    y = xpt1(:, 2); % Extraire les coordonnées Y

    plot(x, y, 'ro'); % 'b' pour bleu et 'o' pour des marqueurs circulaires
    xlabel('X');
    ylabel('Y');
    title(["Affichage de l'image ",num2str(i)," avec les points respectifs"]);
    pause(0.5)

    
end
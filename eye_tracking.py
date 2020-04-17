from __future__ import division
import cv2 as cv
import time
import sys
import array as arr
import os
import statistics
import pyautogui as m
import random
path='./'
#----------------------------------------------------------------------------------------------------------------------------
#Aqui é definida a função que faz a leitura do vídeo e faz a detecção da face do usuário, baseado em uma rede neural profunda.

def detectFaceOpenCVDnn(net, frame):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
            
    return frameOpencvDnn, bboxes
############################################################################################################################
#Neste ponto é criada a função de detecção do olho propriamente dito.
############################################################################################################################
def Detectaolho(imagem_rosto, template_maior, template_menor, i, limite, limiar):
    tentativas=0
    marcador="Found"
    while i<1:
      i=i+1
      if i==1:
        olho = template_maior
        template = cv.cvtColor(olho, cv.COLOR_BGR2GRAY)
        rosto=imagem_rosto
#        limiar=0.7
      if i==2:
        olho = template_menor
        template = cv.cvtColor(olho, cv.COLOR_BGR2GRAY)
        #b1,g1,r1=cv.split(olho)
        #template=b1
        rosto1=imagem_rosto
        rosto=rosto1[B:D, A:C]
#        limiar=0.7
      print(rosto.shape)
      print(olho.shape)

      q=template.shape[0]
      p=template.shape[1]

      cv.imwrite(os.path.join(path,'template.png'),template)


      rosto_gray = cv.cvtColor(rosto, cv.COLOR_BGR2GRAY)

      cv.imwrite(os.path.join(path,'rosto_gray.png'),rosto_gray)

      n=rosto_gray.shape[0]
      m=rosto_gray.shape[1]

      Vetor_E=[]
      X=[]
      Y=[]
      xacum=0
      yacum=0
      Vmax_x=10
      Vmax_y=10
      wmax=1
      wmin=0.7
      Itrmax=100
      c1=0.5	
      c2=0.2
      E=0
      M=m-(p)-1
      N=n-(q)-1
      if M < 1:
        M=1
      if N < 1:
        N=1
      x=random.randrange(0,M,1)
      y=random.randrange(0,N,1)
      j=0
      u=0
      while E < limiar:
        tentativas=tentativas+1
        if j>Itrmax:
          x=random.randrange(0,m-(p)-1,1)
          y=random.randrange(0,n-(q)-1,1)
          if (E<0.4 or u>6):
            Vetor_E=[]
            X=[]
            Y=[]
          xacum=0
          yacum=0
          u=u+1
          E=0
          j=0
          print('tentando novamente pela',u,'ª vez'))
        if j>0:
          Xi=x
          Yi=y
          if j==1:
            Pibest_x=Xi
            Pibest_y=Yi
          else:
            if E>A:
              Pibest_x=Xi
              Pibest_y=Yi
          r1=random.randrange(0,10,1)/10
          r2=random.randrange(0,10,1)/10
          X.append(x)
          Y.append(y)
          Vetor_E.append(E)
          maxE=max(Vetor_E)
          E_pos = Vetor_E.index(maxE)
          Pgbest_x=X[E_pos]
          Pgbest_y=Y[E_pos]
          w=(wmax)-(((wmax-wmin)/Itrmax)*j) #Inércia descendente
          
          if j>1:
            Vit1x=Vitx*w+c1*r1*(Pibest_x-Xi)+c2*r2*(Pgbest_x-Xi)
            Vit1y=Vity*w+c1*r1*(Pibest_y-Yi)+c2*r2*(Pgbest_y-Yi)
          else:
            Vit1x=random.randrange(0,10,1)
            Vit1y=random.randrange(0,10,1)
          if Vit1x>Vmax_x:
            Vit1x=Vmax_x
          if Vit1y>Vmax_y:
            Vit1y=Vmax_y
          Vitx=Vit1x
          Vity=Vit1y
          xacum=xacum+Vit1x
          yacum=yacum+Vit1y
          x=int(round(xacum))
          y=int(round(yacum))
          if x<0:
            x=x*-1
          if y<0:
            y=y*-1
          if x > m-(p)-1:
            x=m-(p)-1
          if y > n-(q)-1:
            y=n-(q)-1
        j=j+1
        T1=0
        I1=0					
        for c in (range(p-1)):
          for t in (range(q-1)):
            I1 = I1 + rosto_gray[(y+t), (x+c)]
            T1 = T1 + template[t, c]
        I2=(1/(p*q))*I1
        T2=(1/(p*q))*T1
        N=0
        D1=0
        D2=0
        for c in (range(p-1)):
          for t in (range(q-1)):
            f_I = (rosto_gray[y+t,x+c] - I2)
            f_T = (template[t,c] - T2)
            N = N + (f_I * f_T)       
            D1 = D1 + (f_I*f_I)
            D2 = D2 + (f_T*f_T)
        A=E
        E = (N)/((D1*D2)**(1/2))
        
        if E > limiar:
          E=round(E*100,2)
          print('match encontrado depois de',u,'tentativas com 100 interações e outras',j,'interações, com',E,'% de certeza')
          time.sleep(3)
          if i==1:
            A=x
            B=y
            C=x+p
            D=y+q
#            rosto_marcado=cv.rectangle(rosto, (x,y), (x+p,y+q), (255, 0, 0), 2)
#            cv.imshow('rosto', rosto_marcado)
#            cv.imwrite(os.path.join(path,'rosto_marcado.png'),rosto_marcado)
            vetor_olho=[A, C, B, D]
          if i==2:
            olho_x1=int(A+x)
            olho_y1=int(B+y)
            olho_x2=int(A+x+p)
            olho_y2=int(B+y+q)
            vetor_olho=[olho_x1,olho_x2,olho_y1,olho_y2]
          k = cv.waitKey(30) & 0xff
          if k==27:
            break
      
        print(E,x,y)
        if tentativas>limite:
          print("olho não encontrado")
          E=limiar
          marcador="NotFound" 
          vetor_olho=0
    return vetor_olho, marcador
#############################################################################################################################
if __name__ == "__main__" :

    #DNN = "TF"
    DNN = "CAFFE"

    if DNN == "CAFFE":
        modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "models/deploy.prototxt"
        net = cv.dnn.readNetFromCaffe(configFile, modelFile)
    else:
        modelFile = "models/opencv_face_detector_uint8.pb"
        configFile = "models/opencv_face_detector.pbtxt"
        net = cv.dnn.readNetFromTensorflow(modelFile, configFile)

    conf_threshold = 0.7

    source = 0
    
    if len(sys.argv) > 1:
        source = sys.argv[1]

    cap = cv.VideoCapture(0)
    
    hasFrame, frame = cap.read()

    frame_count = 0
    tt_opencvDnn = 0
#------------------------------------------------------------------------------------------------
#Nesse ponto é  iniciada efetivamente a detecção do ponto onde o usuário olha na tela.
#O programa usa por base os limites laterais definidos na etapa anterior
#------------------------------------------------------------------------------------------------
#Toda a detecção é feita novamente, com a diferença de que ocorre de maneira indefinida.
    print("Dentro de 2 segundos inciaremos a detecção")
    time.sleep(2)
    contador=0
    cont_localiza=0
    Sx1=3
    Sy1=3
    Sx2=3
    Sy2=3
    template_maior1 = cv.imread('template_maior_direita.png')
    template_maior2 = cv.imread('template_maior_esquerda.png')
    template_menor1 = cv.imread('template_menor_direita.png')
    template_menor2 = cv.imread('template_menor_esquerda.png')
    while (1):
          contador=contador+1  
          hasFrame, frame = cap.read()
          if not hasFrame:
              break
          frame_count += 1

          t = time.time()
          outOpencvDnn, bboxes = detectFaceOpenCVDnn(net,frame)
          y1=bboxes[0][0]
          x1=bboxes[0][1]
          y2=bboxes[0][2]
          x2=bboxes[0][3]
#Retorno da detecção do rosto.
#------------------------------------------------------------------------------------------------
#Feita a detecção dos olhos, com o respectivo filtro necessário.
          img=outOpencvDnn[x1:x2, y1:y2]
          img=cv.medianBlur(img,9)
          cv.imshow('img',img)
          meio=int((y2-y1)/2)
          centro=y1+meio
          centrox=x1+int(3*(x2-x1)/4)
          img11=outOpencvDnn[x1:centrox, y1:centro]
          img21=outOpencvDnn[x1:centrox, centro:y2]
          img1=cv.medianBlur(img11,9)
          img2=cv.medianBlur(img21,9)
          if contador==1:
            vector_eye1, marcador1 = Detectaolho(img1, template_maior1, template_menor1,0, 5000, 0.6)
            vector_eye2, marcador2 = Detectaolho(img2, template_maior2, template_menor2,0, 5000, 0.6)
            time.sleep(5)
            if (marcador1=="NotFound") or (marcador2=="NotFound"):
              contador=0
            if (marcador1=="Found") and (marcador2=="Found"): 
              mem_vector_eye1=vector_eye1
              mem_vector_eye2=vector_eye2
              busca_olho=10
          if contador>1:
            img1_2=img1[int(vector_eye1[2]-Sy1):int(vector_eye1[3]+Sy1), int(vector_eye1[0]-Sx1):int(vector_eye1[1]+Sx1)]
            img2_2=img2[int(vector_eye2[2]-Sy2):int(vector_eye2[3]+Sy2), int(vector_eye2[0]-Sx2):int(vector_eye2[1]+Sx2)]
            vector_eye1, marcador1 = Detectaolho(img1_2, template_maior1, template_menor1,0, busca_olho, 0.1)
            cv.imshow('img1',img1_2)
            vector_eye2, marcador2 = Detectaolho(img2_2, template_maior2, template_menor2,0, busca_olho, 0.1)
            cv.imshow('img2',img2_2)
            if (marcador1=="Found") and (marcador2=="Found"): 
              mem_vector_eye1=vector_eye1
              mem_vector_eye2=vector_eye2
              vector_eye1[0]=vector_eye1[0]+Xt1
              vector_eye1[1]=vector_eye1[1]+Xt1
              vector_eye1[2]=vector_eye1[2]+Yt1
              vector_eye1[3]=vector_eye1[3]+Yt1
              vector_eye2[0]=vector_eye2[0]+Xt2
              vector_eye2[1]=vector_eye2[1]+Xt2
              vector_eye2[2]=vector_eye2[2]+Yt2
              vector_eye2[3]=vector_eye2[3]+Yt2

              dx1=vector_eye1[0]-Xt1
              dx2=vector_eye2[0]-Xt2
              dy1=vector_eye1[2]-Yt1
              dy2=vector_eye2[2]-Yt2
              Sx1=(Sx1+dx1)/contador
              Sx2=(Sx2+dx2)/contador
              Sy1=(Sy1+dy1)/contador
              Sy2=(Sy2+dy2)/contador
              if Sx1 < 2 :
                Sx1=2
              if Sx2 < 2 :
                Sx2=2
              if Sy1 < 2 :
                Sy1=2
              if Sy2 < 2 :
                Sy2=2
            
            if ((marcador1=="NotFound") or (marcador2=="NotFound")):
              vector_eye1=mem_vector_eye1
              vector_eye2=mem_vector_eye2
              cont_localiza=cont_localiza+1
              busca_olho=10
              if cont_localiza>10:
                cont_localiza=0
                print("Olho perdido, procurando novamente.")
                time.sleep(5)
                contador=0
   #           template_maior1=eye1
   #           template_maior2=eye2
                Sx1=4
                Sy1=5
                Sx2=4
                Sy2=5
            if (marcador1=="Found") and (marcador2=="Found") and (cont_localiza>0):
              cont_localiza=0
          if (marcador1=="Found") and (marcador2=="Found"):    
            Xt1=vector_eye1[0]
            Xt2=vector_eye2[0]     
            Yt1=vector_eye1[2]
            Yt2=vector_eye2[2]  
            print("vetor olho1", vector_eye1[2], vector_eye1[3], vector_eye1[0], vector_eye1[1]) 
            print("vetor olho2", vector_eye2[2], vector_eye2[3], vector_eye2[0], vector_eye2[1])
            eye1=img1[vector_eye1[2]:vector_eye1[3], vector_eye1[0]:vector_eye1[1]]
            eye2=img2[vector_eye2[2]:vector_eye2[3], vector_eye2[0]:vector_eye2[1]]
            cv.imshow('img1',eye1)
            cv.imshow('img2',eye2)

#-------------------------------------------------------------------------------------------------------------------    
          if frame_count == 1:
              tt_opencvDnn = 0

          k = cv.waitKey(10)
          if k == 27:
              break
              cv2.destroyAllWindows()

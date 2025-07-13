/*invariant*/
int NUM=100;      /*number of particle*/
int Size=300;     /*domain side length*/
int effectNUM=6;  /*topological rank*/
float nyu=0.50;   /*Noise paramater*/
float Q=1.00;     /*Interaction paramater*/
float R=100;      /*metric range*/
float alpha = 1;  /*distance-weight parameter*/
long RS = 1;      /*random seed*/

/*non-invariant*/
float va,Va=0,aveVa=0;
ArrayList<Particle>particle=new ArrayList<Particle>();
PVector[] new_vel = new PVector[NUM];
PVector velMT = new PVector(0,0,0);
String[] saveData = new String[101];
String saveFileName;
int c=0;
int z=0;


void setup(){
 size(1000,600,P3D);
 background(0);
 noFill();
 stroke(255);
 strokeWeight(2);
 smooth();
 frameRate(200);
 textSize(15);
 randomSeed(RS);
 for(int i=0;i<NUM;i++){
  particle.add(new Particle(random(-Size/2,Size/2),random(-Size/2,Size/2),random(-Size/2,Size/2)));
 }
  for (int i = 0; i < NUM; i++) 
  {
    Particle p = particle.get(i);
    p.no = i;
    Particle other = particle.get(i);
    other.no = i;
  }

}

void draw(){
   background(0);
   orderparamater();
   countNUM();
   time();
   graph();
   translate(width/3,height/2);
   
   scale(0.5); 
   
   rotateX(map(mouseY, 0, height, HALF_PI, -HALF_PI));
   rotateY(map(mouseX, 0, width, -HALF_PI, HALF_PI));
   strokeWeight(10);
   box(Size);
   strokeWeight(2);
    for(int i=0;i<particle.size();i++){
    Particle p=particle.get(i);
    p.update();
    new_vel[p.no]=new PVector(velMT.x,velMT.y,velMT.z);
   }
    for(int i=0;i<particle.size();i++){
     Particle pa=particle.get(i);
     pa.velocity=new PVector(new_vel[i].x,new_vel[i].y,new_vel[i].z); 
     pa.updatePosition();
     pa.distance();
     pa.display();
    }
   if(frameCount==1||frameCount%100==0)
   {
     println(va);
     saveData[z] = str(va);
     z++;
   }
   if(frameCount==10001)noLoop();
   if(!(va>0)){
    println(frameCount);
    noLoop();
   }

   saveFileName = makeFileName(Q, nyu, RS);
   saveStrings(saveFileName, saveData);
   //saveFrame("frames/frame-####.png");/*save each frame→ tarminal : "ffmpeg -framerate 60 -i frames/frame-%04d.png -c:v libx264 -pix_fmt yuv420p simulation.mp4" */

}


class Particle{
  int no;
  float x,y,z;
  float theta=random(TWO_PI),gamma=random(TWO_PI);
  PVector position;
  PVector velocity;
  PVector velT,velM;
  int rast=0;
  
 Particle(float x, float y, float z){
  position = new PVector(x,y,z);
  velocity = new PVector(sin(theta)*cos(gamma),sin(theta)*sin(gamma),cos(theta));
 }
 
  void distance(){
   if(position.x<-Size/2) position.x=Size/2;
   if(position.y<-Size/2) position.y=Size/2;
   if(position.z<-Size/2) position.z=Size/2;
   if(position.x>Size/2) position.x=-Size/2;
   if(position.y>Size/2) position.y=-Size/2;
   if(position.z>Size/2) position.z=-Size/2;
  }
 
  void display(){ 
   strokeWeight(8);
   point(position.x,position.y,position.z); 
   strokeWeight(1);
   line(position.x,position.y,position.z
       ,position.x+30*velocity.x/2,position.y+30*velocity.y/2,position.z+30*velocity.z/2);
  }

  void updatePosition(){
   position.add(velocity);
  }
  
  PVector update(){
   PVector sum = new PVector(velocity.x,velocity.y,velocity.z);
   float count=1,countT=0;
   PVector[] othpos = new PVector[7];
   PVector vel,avaV,N,e,p,pv,M;
   float Xi=nyu*random(-PI,PI);
   velMT=new PVector(0,0,0);
    float [] d_othpos = new float[7];
    float [] d = new float[NUM];
    int[] min_index = new int[NUM];

   for(Particle other : particle){
    othpos[0]=new PVector(other.position.x+Size,other.position.y,other.position.z);
    othpos[1]=new PVector(other.position.x,other.position.y+Size,other.position.z);
    othpos[2]=new PVector(other.position.x,other.position.y,other.position.z+Size);
    othpos[3]=new PVector(other.position.x-Size,other.position.y,other.position.z);
    othpos[4]=new PVector(other.position.x,other.position.y-Size,other.position.z);
    othpos[5]=new PVector(other.position.x,other.position.y,other.position.z-Size);
    othpos[6]=new PVector(other.position.x,other.position.y,other.position.z);
    float d1 = Size+1;
    for(int i=0;i<othpos.length;i++){
     float otherd = PVector.dist(position,othpos[i]);
      if(d1>otherd){
       d1 = otherd;
      }
    }
    if((d1>0)&&(d1<R)){
      vel = new PVector(other.velocity.x,other.velocity.y,other.velocity.z);
      float dWeightM = pow(d1,alpha);
      PVector velWei=PVector.div(vel, dWeightM);
      sum.add(velWei);
      count++;
    }
      if (other.no == no)
      {
        d[no] = 0;
      } 
      else if (other.no != no)
      {
        ArrayList<String> d_othpos_list;
        d_othpos_list = new ArrayList<String>();

        for (int i = 0; i < 7; i++)
        {
          d_othpos[i] = PVector.dist(position, othpos[i]);
          d_othpos_list.add(str(d_othpos[i]));
        }

        min_index[other.no] = d_othpos_list.indexOf(str(min(d_othpos)));
        d[other.no] = d1;
      }
    }


    ArrayList<String> d_list;
    d_list = new ArrayList<String>();
    for (int i = 0; i < NUM; i++) d_list.add(str(d[i]));
    d = sort(d);
    
    PVector sumN = new PVector (velocity.x,velocity.y,velocity.z);

    PVector avaVN,NN,eN,pN,pvN,MN;
    float dWeight;

    for (int i = 0; i <= effectNUM; i++)
    {
      int j = d_list.indexOf(str(d[i]));
      Particle q = particle.get(j);
      if (i == 0){
       dWeight = 1;
      }
      else {
       dWeight = pow(d[i],alpha);
      }
      PVector q1=PVector.div(q.velocity, dWeight);
      sumN.add(q1);
      countT++;
    }
    avaV=sum.div(count);
    N=avaV.div(avaV.mag());
    p = new PVector(random(0,1),random(0,1),0);
    pv = new PVector(p.x,p.y,(-N.x*p.x-N.y*p.y)/N.z);
    e=pv.normalize();
    M=new PVector(f11(e,Xi)+f12(e,Xi)+f13(e,Xi)
                 ,f21(e,Xi)+f22(e,Xi)+f23(e,Xi)
                 ,f31(e,Xi)+f32(e,Xi)+f33(e,Xi));
    
    PVector velocityM=new PVector(N.x,N.y,N.z);
    velM = new PVector(velocityM.x*M.x,velocityM.y*M.y,velocityM.z*M.z);
    
    avaVN=sumN.div(countT);
    NN=avaVN.div(avaVN.mag());
    pN = new PVector(random(0,1),random(0,1),0);
    pvN = new PVector(pN.x,pN.y,(-NN.x*pN.x-NN.y*pN.y)/NN.z);
    eN=pvN.normalize();
    MN=new PVector(f11(eN,Xi)+f12(eN,Xi)+f13(eN,Xi)
                 ,f21(eN,Xi)+f22(eN,Xi)+f23(eN,Xi)
                 ,f31(eN,Xi)+f32(eN,Xi)+f33(eN,Xi));
    
    PVector velocityT=new PVector(NN.x,NN.y,NN.z);
    velT = new PVector(velocityT.x*MN.x,velocityT.y*MN.y,velocityT.z*MN.z);
    
    velM.mult(1-Q);
    velT.mult(Q);
    velMT.add(velM);
    velMT.add(velT);
    velMT.normalize();
    
    return velMT;
   }

}/*class Particle end*/
void mousePressed(){
 particle.add(new Particle(random(Size),random(Size),random(Size)));
}

void time(){
   text("order parameter",width/2+10,200);
   
   text("t",width/2+10,250);
  
   text(frameCount,width/2+10+105,250);

   String textnoise = "η = " + nyu;
   text(textnoise,width/2+10,300);
   
   String textQ = "Q = " + Q;
   text(textQ,width/2+10,330);
   
   String textSize = "L = " + Size;
   text(textSize,1*width/2+10,360);
}

void graph(){
   pushMatrix();
   noFill();
   translate(width/2+10+width/6-5,220);
   box(width/3,20,0);
   popMatrix();
   
   pushMatrix();
   translate(width/2+10+width*va/6-5,220);
   fill(211,207,217);
   box(width*va/3,20,0);
   noFill();
   popMatrix();
   
   pushMatrix();
   translate(width/2+10+width*aveVa/3-5,220);
   stroke(220,20,60);
   box(1,20,0);
   stroke(255);
   popMatrix();
}

void countNUM(){
 int n=particle.size();
 String textNum = "N = " + n;
 text(textNum,width/2+10,390);
}

void orderparamater(){
 PVector total_vel = new PVector(0,0,0);
 for(int i=0;i<particle.size();i++){
  Particle p = particle.get(i);
  total_vel.add(p.velocity);
 }
 float  m = total_vel.mag();
 va = m/particle.size();
}

/*Rodligues's rotation formula*/
float f11(PVector n, float theta){
 return n.x*n.x*(1-cos(theta))+cos(theta);
}
float f12(PVector n,float theta){
 return n.x*n.y*(1-cos(theta))-n.z*sin(theta);
}
float f13(PVector n,float theta){
 return n.x*n.z*(1-cos(theta))+n.y*sin(theta);
}
float f21(PVector n,float theta){
 return n.x*n.y*(1-cos(theta))+n.z*sin(theta);
}
float f22(PVector n,float theta){
 return n.y*n.y*(1-cos(theta))+cos(theta);
}
float f23(PVector n,float theta){
 return n.y*n.z*(1-cos(theta))-n.x*sin(theta);
}
float f31(PVector n,float theta){
 return n.x*n.z*(1-cos(theta))-n.y*sin(theta);
}
float f32(PVector n,float theta){
 return n.y*n.z*(1-cos(theta))+n.x*sin(theta);
}
float f33(PVector n,float theta){
 return n.z*n.z*(1-cos(theta))+cos(theta);
}

   String makeFileName(float Qval, float nyuVal, long seedVal) {
  /* nf(value, left, right): 桁数そろえ＆小数点以下 right 桁 */
  String qStr   = nf(Qval,   0, 2);  // 1 桁小数
  String nyuStr = nf(nyuVal, 0, 2);  
  return + Size + "cloop(" + qStr + "," + nyuStr + "," + seedVal + ").txt";
   }

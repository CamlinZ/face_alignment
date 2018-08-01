import os
import json
import shutil
import MySQLdb

db_url = "172.16.32.9"
db_username = "face_db_user"
db_pwd = "db_recognition"
db_schame = "db_face_recognition"
charset_l = 'utf8'

image_dir = '/data/face_images'
assert os.path.exists(image_dir)
save_dir = '/data/face_badcase_bybychen'
if not os.path.exists(save_dir):
	os.mkdir(save_dir)
check_undo = False	

def copy_im(save_dir, image_dir, result, undone):
	for item in result:
		im1_p =  item["Fimage1_path"]
		im2_p = item["Fimage2_path"]
		im_id = item["Fidcard_path"]
		im_score1 = item["Fimage1_score"]
		im_score2 = item["Fimage2_score"]
		id_name = item["Fidcard_name"]
		
		
		if os.path.exists(os.path.join(image_dir,im1_p)):
			dir_name = os.path.join(save_dir,id_name)
			idcard=os.path.basename(im_id)
			if not os.path.exists(dir_name):
				os.mkdir(dir_name)
			try:	
				###fetch images
				shutil.copy(os.path.join(image_dir,im1_p), os.path.join(dir_name,idcard+'_1.jpg'))
			except:
				os.remove(dir_name)
				print('CAN NOT COPY THE IMAGE')
			else:
				print('trans image')
				shutil.copy(os.path.join(image_dir,im2_p),os.path.join(dir_name,idcard+'_2.jpg'))
				shutil.copy(os.path.join(image_dir,im_id), os.path.join(dir_name,idcard+'_id.jpg'))
				
				
				score_str = 'score_'+str(im_score1)+'_'+str(im_score2)+'.txt'
				print(score_str)
				f = open(os.path.join(dir_name,score_str),'w')
				f.write("")
				f.close
		else:
			undone.append(item)

    return undone
	

dbcon = MySQLdb.Connect(host=db_url,user=db_username,passwd=db_pwd,db=db_schame,charset=charset_l,port=3306)
cur = dbcon.cursor(cursorclass=MySQLdb.cursors.DictCursor)
select = "select Fidcard_name,Fimage1_path, Fimage2_path, Fidcard_path, Fimage1_score , Fimage2_score from t_face_verification where Fappid='CuyWK31tyzOTwD6Z8CvAwfi3PedXqnf0' and Ffinal_status=0 and Fclient_ip!='14.17.22.31' "
cur.execute(select)
result = cur.fetchall()

if check_undo:
	whole_file = open(save_dir+'/whole.json','r')
	whole_list = json.loads(whole_file.read())
	whole_file.close()
else:
	whole_list=[]

	
for i in result:
	whole_list.append(i)
whole_file = open(save_dir+'/whole.json','w')
whole_file.write(json.dumps(whole_list))	
	


undo_file = open(save_dir+'/unfinished.json','w')	
undone=[]


undone = copy_im(save_dir, image_dir, whole_list, undone)
print('total result',str(len(whole_list)))
print('undone item',str(len(undone)))
if len(undone)!=0:
	undo_file.write(json.dumps(undone))
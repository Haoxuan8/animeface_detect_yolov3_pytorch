# -*- coding: utf-8 -*-
"""
"""

import urllib.request
import os,re,requests

class PixivSpider:
    def __init__(self):
        self.headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134',
                      'Referer':''}
        self.proxies = {'http': 'http://127.0.0.1:1080', 'https': 'http://127.0.0.1:1080'}  
        self.id_set=set()
    
    def getid(self,url):
        pixiv_id=re.search('\d{7,9}',url).group()
        return pixiv_id
    
    def getReferer(self,pixiv_id):
        referer='https://www.pixiv.net/member_illust.php?mode=medium&illust_id='+pixiv_id
        return referer
    
    def gethtml(self,url):
        self.headers['Referer']=''
        page=requests.get(url,headers=self.headers,proxies=self.proxies)
        html=page.content.decode('utf-8')
        return html
    
    def save_pixiv_imgs(self,urls,path):
        for url in urls:
            pixiv_id=self.getid(url)
            if pixiv_id in self.id_set:
                print('image id:%s was already saved'%pixiv_id)
                continue
            self.id_set.add(pixiv_id)
            self.headers['Referer']=self.getReferer(pixiv_id)
            img=requests.get(url,headers=self.headers)
            img_format='.jpg'
            if img.status_code==404:
                url=url.replace('.jpg','.png')
                img=requests.get(url,headers=self.headers)
                img_format='.png'
            
            savepath=path+'/%s%s'%(pixiv_id,img_format)
            with open(savepath,'wb') as file:
                file.write(img.content)
    
    
    def get_pixiv_imgs(self,date,page):
        """
        @param date: 20190530
        each page has 50 images
        """
        path='./pixiv_imgs/'+date
        if not os.path.exists(path):
            os.makedirs(path)
            print("make dir done.")
        else:
            print("dir already exsits.")
        urls=[]
        
        for p in range(1,page+1):
            url='https://www.pixiv.net/ranking.php?mode=daily&content=illust&p=%s&date=%s'%(p,date)
            ranking_html=self.gethtml(url)
            imgs_soup=re.findall(r'data-filter="thumbnail-filter lazy-image"data-src="(.+?\.jpg)"',ranking_html)
            for img_url in imgs_soup:
                url_=img_url.replace('c/240x480/img-master','img-original')
                url_=url_.replace('_master1200','')
                urls.append(url_)
            
        print("saving %s ranking images..."%date)
        self.save_pixiv_imgs(urls,path)
        
            
if __name__=='__main__':
   spider=PixivSpider()
   date=20190501
   while date!=20190532:
       spider.get_pixiv_imgs("%s"%date,4)
       date+=1
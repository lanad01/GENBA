import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

class Crawling:
    def __init__(self, keyword):
        """
        Crawling 초기화.

        Args:
            keyword (str): 검색할 키워드.
        """
        self.keyword = keyword

    def makePgNum(self, num):
        """
        페이지 번호를 URL 형식에 맞게 변환.

        Args:
            num (int): 페이지 번호.

        Returns:
            int: URL에 사용할 페이지 번호.
        """
        if num == 1:
            return num
        elif num == 0:
            return num + 1
        else:
            return num + 9 * (num - 1)

    def makeUrl(self, search, start_pg, end_pg):
        """
        URL 목록 생성.

        Args:
            search (str): 검색어.
            start_pg (int): 시작 페이지 번호.
            end_pg (int): 끝 페이지 번호.

        Returns:
            list: 생성된 URL 목록.
        """
        urls = []
        for i in range(start_pg, end_pg + 1):
            page = self.makePgNum(i)
            url = f"https://search.naver.com/search.naver?where=news&sm=tab_pge&query={search}&start={page}"
            urls.append(url)
        # print("생성된 URL: ", urls)
        return urls

    def news_attrs_crawler(self, articles, attrs):
        """
        HTML 요소에서 특정 속성 값 추출.

        Args:
            articles (list): HTML 요소 목록.
            attrs (str): 추출할 속성 이름.

        Returns:
            list: 추출된 속성 값.
        """
        attrs_content = []
        for i in articles:
            attrs_content.append(i.attrs[attrs])
        return attrs_content

    def articles_crawler(self, url):
        """
        기사 링크 크롤링.

        Args:
            url (str): 대상 URL.

        Returns:
            list: 추출된 기사 링크.
        """
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"}
        original_html = requests.get(url, headers=headers)
        html = BeautifulSoup(original_html.text, "html.parser")

        url_naver = html.select("div.group_news > ul.list_news > li div.news_area > div.news_info > div.info_group > a.info")
        return self.news_attrs_crawler(url_naver, 'href')
        
    def run(self, start_pg=None, end_pg=None):
        """
        크롤링 실행.

        Args:
            start_pg (int, optional): 시작 페이지 번호 (기본값: 1).
            end_pg (int, optional): 끝 페이지 번호 (기본값: 10).

        Returns:
            list: 최종 NAVER 뉴스 링크 목록.
        """
        # 기본값 설정
        start_pg = start_pg or 1
        end_pg = end_pg or 10

        urls = self.makeUrl(self.keyword, start_pg, end_pg)

        news_urls = []
        for url in urls:
            links = self.articles_crawler(url)
            news_urls.extend(links)

        # NAVER 뉴스만 필터링
        final_urls = [x for x in news_urls if 'news.naver.com' in x]
        
        list_news_content = []
        list_news_titles  = []
        list_news_dates   = []
        list_news_section = []
        list_news_url = []
        
        for i, url in enumerate(final_urls) :
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                news_content = soup.find('div', {'id': 'newsct_article'})
                news_titles = soup.find('h2', {'id': 'title_area'})
                news_dates  = soup.find('span', {'class': 'media_end_head_info_datestamp_time'})
                news_section= soup.find('em', {'class': 'media_end_categorize_item'})
                  
                # 본문에서 스크립트 제거 및 리스트에 추가
                if news_content:
                    for script in news_content(["script", "style"]):
                        script.decompose()
                    list_news_content.append(news_content.text.strip())
                else :
                    list_news_content.append(" ")
                    
                if news_titles:
                    for script in news_titles(["script", "style"]):
                        script.decompose()
                    list_news_titles.append(news_titles.text.strip())
                else :
                    list_news_titles.append(" ")
                    
                if news_dates:
                    for script in news_dates(["script", "style"]):
                        script.decompose()
                    list_news_dates.append(news_dates.text.strip())
                else :
                    list_news_dates.append(" ")
                    
                if news_section:
                    for script in news_section(["script", "style"]):
                        script.decompose()
                    list_news_section.append(news_section.text.strip())
                else :
                    list_news_section.append(" ")
                
                list_news_url.append(url)
        
        df_result = pd.DataFrame({
              'title'   : list_news_titles 
            , 'content' : list_news_content
            , 'dates'   : list_news_dates  
            , 'section' : list_news_section
            , 'url'     : list_news_url
        })
        previous_day = (datetime.now() - timedelta(days=1)).strftime('%Y.%m.%d')
        df_result = df_result[df_result['dates'].str[:10] == previous_day].reset_index(drop=True)
            
        return df_result

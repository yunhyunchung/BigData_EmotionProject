''' 빅데이터 프로젝트: 감정대화 말뭉치 분석 '''
    
''' 문제 1-1: 고민 상담 발화에서 전체적으로 어떤 단어가 많이 나타나는가?  '''

import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

# 데이터셋 로드
df = pd.read_csv("감성대화말뭉치(최종데이터)_Training.csv")

# 한글로 표시하기 위해 => 한글 폰트인 맑은고딕체 설정
font_path = "c:/Windows/fonts/malgun.ttf"
font_name = fm.FontProperties(fname = font_path).get_name()
plt.rc('font', family=font_name)

# Okt 형태소 분석기 초기화
okt = Okt()

# 사람문장1에서 명사 추출
def get_nouns(text):
    nouns = okt.nouns(text)
    return ' '.join(nouns)

df['nouns'] = df['사람문장1'].apply(get_nouns)
df['nouns']

# 불용어 리스트 (추출한 명사 중에서 제외)
stopwords = ['정말', '때문', '우리', '이번', '이제', '화가', '자꾸', '지금', '다른', '자신',
             '계속', '어제', '갑자기', '아무', '하나', '보고', '얼마', '항상', '점점', '수가',
             '모두', '요새', '진짜', '대해', '달라', '위해', '사실', '자기', '가지', '최근', 
             '이유', '벌써', '자주', '예전', '다리', '제대로', '모든', '이상', '모습', '내일', 
             '언제', '조금', '해도', '아주', '무슨', '가기', '먼저', '일도', '매우', '제일']

# TfidfVectorizer를 사용하여 DTM 생성 (TF-IDF 행렬)
vectorizer = TfidfVectorizer(stop_words=stopwords)
dtm = vectorizer.fit_transform(df['nouns'])
print(dtm)

# 단어 사전 구성
word_list = vectorizer.get_feature_names_out()
tfidf_scores = dtm.sum(axis=0).tolist()[0]
word_dict = dict(zip(word_list, tfidf_scores))

# 단어 개수 기준으로 내림차순 정렬
sorted_word_dict = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)

# 바 차트 그리기
top_n = 20  # 상위 N개 단어만 표시
top_words = [word for word, score in sorted_word_dict[:top_n]]
top_scores = [score for word, score in sorted_word_dict[:top_n]]

plt.figure(figsize=(10, 6))
plt.bar(top_words, top_scores)
plt.xlabel('단어')
plt.ylabel('TF-IDF 점수')
plt.title('전체 고민 상담 발화 중 상위 {}개 단어 TF-IDF 점수'.format(top_n))
plt.xticks(rotation=0)
plt.show()


# In[]:
    
''' 문제 1-2. 전체적으로 많이 등장하는 단어 10개에서 어떤 감정이 가장 많이 나타나는가? '''

''' 1) 상위 10개 단어에서 6개 감정_대분류별로 상위 3개 감정_소분류를 scatter plot으로 시각화 '''
import seaborn as sns

# 상위 10개 단어에서 많이 나타나는 감정을 각 감정_대분류별로 3개의 감정_소분류 추출
emotion_n = 10
top_words = [word for word, score in sorted_word_dict[:emotion_n]]

# 감정 대분류와 소분류별 개수 계산
emotion_data = df[df['nouns'].str.contains('|'.join(top_words))]
emotion_counts = emotion_data.groupby(['감정_대분류', '감정_소분류']).size().reset_index(name='count')
emotion_counts = emotion_counts.sort_values(by='count', ascending=False).groupby('감정_대분류').head(3)

# scatter plot 시각화 (점의 크기는 해당 감정의 빈도 수.)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=emotion_counts, x='감정_대분류', y='감정_소분류', size='count', hue='count', sizes=(50, 300), palette='cool', alpha=0.8, legend=False)
plt.xlabel('감정 대분류')
plt.ylabel('감정 소분류')
plt.title('상위 {}개 단어에서 많이 나타나는 감정_대분류별 3개 감정_소분류'.format(emotion_n))
plt.show()


''' 2) 상위 10개 단어에 대해 1등 ~ 10등 각 단어에서 가장 많이 나타나는 상위 5개 감정을 Pie chart로 시각화 '''

# 10개의 그래프를 표현하기 위한 subplot 설정
fig, axes = plt.subplots(5, 2, figsize=(16, 20))

for word_rank in range(10):
    word = sorted_word_dict[word_rank][0]
    row = word_rank // 2
    col = word_rank % 2

    word_emotion = df[df['nouns'].str.contains(word)]
    word_emotion_counts = word_emotion.groupby(['감정_대분류', '감정_소분류']).size().reset_index(name='count')
    word_emotion_counts = word_emotion_counts.sort_values(by='count', ascending=False).head(5)

    # 시각화 - pie 차트
    colors = plt.cm.get_cmap('Pastel1')(range(len(word_emotion_counts)))
    axes[row, col].pie(word_emotion_counts['count'], labels=word_emotion_counts['감정_소분류'], colors=colors,
                       autopct='%1.1f%%', startangle=90)
    axes[row, col].set_title(f'{word_rank+1}등 Word: {word}')
    axes[row, col].axis('equal')

plt.suptitle("1등부터 10등 단어에서 가장 많이 등장하는 감정 5개 (Pie 차트)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# In[]:
    
''' 문제 2-3. 연령별과 성별로 고민 상담 발화에서 어떤 단어가 많이 나타나는가? '''
''' TfidfVectorizer를 사용하여 연령별과 성별로 그룹화된 단어 사전을 구성하고, 상위 단어의 TF-IDF 점수가 높은 순서대로 시각화한다. '''

from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# 연령별 그룹화하여 단어 사전 구성
age_groups = df.groupby('연령')
age_word_dict = {}
for age, group in age_groups:
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    dtm = vectorizer.fit_transform(group['nouns'])
    word_list = vectorizer.get_feature_names_out()
    tfidf_list = dtm.sum(axis=0).tolist()[0]
    word_dict = dict(zip(word_list, tfidf_list))
    sorted_word_dict = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
    age_word_dict[age] = sorted_word_dict

# 성별 그룹화하여 단어 사전 구성
gender_groups = df.groupby('성별')
gender_word_dict = {}
for gender, group in gender_groups:
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    dtm = vectorizer.fit_transform(group['nouns'])
    word_list = vectorizer.get_feature_names_out()
    tfidf_list = dtm.sum(axis=0).tolist()[0]
    word_dict = dict(zip(word_list, tfidf_list))
    sorted_word_dict = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
    gender_word_dict[gender] = sorted_word_dict


# 상위 N개 단어만 표시
top_n = 20  


''' 1) 연령별, 성별로 많이 사용하는 단어를 WordCloud 시각화 '''
def generate_wordcloud(data):
    wordcloud = WordCloud(width=800, height=400, font_path=font_path, background_color='white').generate_from_frequencies(data)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# 연령별 상위 단어 Word Cloud 그리기
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, (age, word_dict) in enumerate(age_word_dict.items()):
    top_words = {word: tfidf for word, tfidf in word_dict[:top_n]}
    title = '연령별 상위 {}개 단어 Word Cloud ({}대)'.format(top_n, age)
    wordcloud = WordCloud(width=400, height=200, background_color='white', font_path=font_path).generate_from_frequencies(top_words)
    axes[i // 2, i % 2].imshow(wordcloud, interpolation='bilinear')
    axes[i // 2, i % 2].set_title(title)
    axes[i // 2, i % 2].axis('off')

plt.tight_layout()
plt.show()

# 성별 상위 단어 Word Cloud 그리기
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, (gender, word_dict) in enumerate(gender_word_dict.items()):
    top_words = {word: tfidf for word, tfidf in word_dict[:top_n]}
    title = '성별 상위 {}개 단어 Word Cloud ({})'.format(top_n, gender)
    wordcloud = WordCloud(width=400, height=200, background_color='white', font_path=font_path).generate_from_frequencies(top_words)
    axes[i].imshow(wordcloud, interpolation='bilinear')
    axes[i].set_title(title)
    axes[i].axis('off')

plt.tight_layout()
plt.show()


''' 2) 연령별, 성별로 많이 사용하는 단어를 Bar 차트 그리기 '''
# 색상 리스트 생성
color_palette = sns.color_palette("summer", n_colors=top_n)

# 연령별 상위 단어 바 차트 그리기
plt.figure(figsize=(14, 10))
for i, (age, word_dict) in enumerate(age_word_dict.items()):
    top_words = [word for word, tfidf in word_dict[:top_n]]
    top_tfidf = [tfidf for word, tfidf in word_dict[:top_n]]
    plt.subplot(2, 2, i+1)
    plt.bar(top_words, top_tfidf, color=color_palette)
    plt.xlabel('단어')
    plt.ylabel('TF-IDF')
    plt.title('연령별 상위 {}개 단어 TF-IDF ({}대)'.format(top_n, age))
    plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 성별 상위 단어 바 차트 그리기
plt.figure(figsize=(14, 5))
for i, (gender, word_dict) in enumerate(gender_word_dict.items()):
    top_words = [word for word, tfidf in word_dict[:top_n]]
    top_tfidf = [tfidf for word, tfidf in word_dict[:top_n]]
    plt.subplot(1, 2, i+1)
    plt.bar(top_words, top_tfidf, color=color_palette)
    plt.xlabel('단어')
    plt.ylabel('TF-IDF')
    plt.title('성별 상위 {}개 단어 TF-IDF ({})'.format(top_n, gender))
    plt.xticks(rotation=15)
plt.tight_layout()
plt.show()








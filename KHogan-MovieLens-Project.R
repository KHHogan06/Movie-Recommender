## HarvardX Data Science Capstone Project: Movie Recommendation System
## Author: Karen H. Hogan

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require("viridis")) install.packages("viridis")
if(!require(knitr)) install.packages("knitr")
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(recosystem)) install.packages("recosystem")


# Load required libraries
library(tidyverse)
library(caret)
library(data.table)
library(gridExtra) 
library(lubridate)
library("viridis")
library(knitr)
library(kableExtra)
library(recosystem)

options(scipen = 999, digits = 6)



###############################################################
####          Create edx set, validation set               ####

# Code provided by course instructions
# Note: this process could take a couple of minutes

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# Using code for R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



###############################################################
####               Exploring edx dataset                   ####


# Summarizing dataset
edx %>% summarize(users = n_distinct(userId), movies = n_distinct(movieId), observations = n(), average = mean(rating))

# print lines
knitr::kable(head(edx[, 1:6]), "simple")




# Ratings 
edx %>% group_by(rating) %>% summarize(n = n()) %>%
  ggplot(aes(rating, n)) + 
  geom_bar(stat = "identity", fill = "slategrey") + 
  geom_text(aes(label = n), size = 2, vjust = -.5, alpha = 3/4) + labs(x="Ratings",y="") + 
  geom_vline(xintercept = mean(edx$rating), color = "blue", linetype = "dashed", size = 1) +  
  annotate("text", x = 2.8, y = 2555000, label = "Average Rating: 3.51", color = "blue", size = 3) + theme_bw(base_size = 9) + 
  theme(axis.text.y=element_blank(), axis.ticks.y = element_blank(), plot.title = element_text(size = 10)) + 
  ggtitle("Ratings in Dataset")


# Average movie rating vs # of ratings. Movies with more ratings tend to have higher ratings
edx %>% group_by(movieId) %>% summarize(avg = mean(rating), n = n()) %>% 
  ggplot(aes(avg, n, size = n)) +
  geom_point(alpha = 1/5, color = "blue") + 
  xlab("Average Rating of Movie") + ylab("") + theme_bw(base_size=9) + 
  theme(plot.title = element_text(size = 10), legend.position="bottom") + 
  ggtitle("Average Movie Rating by Number of Ratings")

# Average movie ratings
edx %>% group_by(title) %>% summarize(avg = mean(rating), n = n()) %>% 
  ggplot(aes(title, avg, size = n)) +
  geom_point(alpha = 1/5, color = "blue") +  
  ylab("Ratings") + scale_x_discrete(name = "Movies", labels = NULL) + 
  theme(text=element_text(size=9), legend.position = "none") + 
  ggtitle("Average Ratings by Movie")

# Average user ratings
edx %>% group_by(userId) %>% summarize(avg = mean(rating), n = n()) %>% 
  ggplot(aes(userId, avg, size = n)) +
  geom_point(alpha = 1/10, color = "blue") +
  theme_bw(base_size=9) +labs(x = "Users", y = "Ratings") + 
  theme(panel.grid=element_blank(), legend.position = "none") + 
  ggtitle("Average Rating by User")



edx %>% group_by(movieId) %>% summarize( avg = mean(rating)) %>% summarize(overall = mean(avg))
edx %>% group_by(userId) %>% summarize( avg = mean(rating)) %>% summarize(overall = mean(avg))




###############################################################
##          Extracting date features for analysis           ##

# Converting timestamp to date, separating year from title
edx1 <- edx %>% mutate(date_rated = as_datetime(timestamp), year = str_extract(title, "\\(\\d{4}\\)") %>% str_remove_all("[\\(\\)]"),
                      title = str_remove(title, "\\(\\d{4}\\)$") %>% str_trim())

# Removing hhmmss from date_rated & timestamp
edx1 <- edx1 %>% mutate(date_rated = date(date_rated)) %>% select(-timestamp)
# Calculating age(in years) & by week of movie when rated & converting year to numeric
edx1 <- edx1 %>% mutate(age_rated = as.numeric(year(date_rated)) - as.numeric(year), year = as.numeric(year), week_rated = round_date(date_rated, unit = "week")) 

# Check new age_rated column
sum(edx1$age_rated < 0)
# Found -1 age_rated values when timestamp was before year in title. Updated to 0.
edx1$age_rated[edx1$age_rated < 0] <- 0

# Removing original & large edx
rm(edx)


###############################################################
##       Exploring age of movie & age when rated effect      ##  

# avg by year released
edx1 %>% group_by(year) %>% 
  summarize(avg = mean(rating), n = n()) %>%
  ggplot(aes(year, avg, size = n)) + ylim(2,4.5) +
  geom_point(alpha = 1/4) + geom_smooth() + scale_x_reverse() +
  theme_bw() + ylab("rating") + xlab("year released") +
  theme(legend.position = "none",) + ggtitle("Average Rating by Year Released")

# avg by age when rated
edx1 %>% group_by(age_rated) %>% 
  summarize(avg = mean(rating), n = n()) %>%
  ggplot(aes(age_rated, avg, size = n)) + ylim(2,4.5) +
  geom_point(alpha = 1/3) + geom_smooth(model = glm) +
  theme_bw() + theme(legend.position = "none") + ylab("") + xlab("age when rated") +
  ggtitle("Rating by Age of Movie When Rated")



# avg by date rated
edx1 %>% group_by(date_rated) %>% summarize(avg = mean(rating), n = n()) %>%
  ggplot(aes(date_rated, avg, size = n)) +
  geom_point(alpha = 1/4) + geom_smooth() + ylim(2,4.5) +
  theme_bw() + xlab("rating by week rated") +
  theme(legend.position = "none") + ggtitle("Average Rating by Date Rated")

# avg by week rated. (days with missing values. rounding to week)
edx1 %>% group_by(week_rated) %>% summarize(avg = mean(rating), n = n()) %>%
  ggplot(aes(week_rated, avg, size = n)) +
  geom_point(alpha = 1/4) + geom_smooth() + ylim(2,4.5) +
  theme_bw() + xlab("rating by week rated") +
  theme(legend.position = "none") + ggtitle("Average Rating by Week of Year Rated")



# avg by number of days since user started rating movies
edx1 %>% group_by(userId) %>% mutate(days_rated = as.numeric(date_rated - min(date_rated))) %>%
  ungroup() %>% group_by(days_rated) %>% summarize(avg = mean(rating), n = n()) %>%
  ggplot(aes(days_rated, avg, size = n)) + 
  geom_point(alpha = 1/5) +
  labs(x = "days since first rating", y = "ratings") + theme_bw() + 
  theme(legend.position = "none") + ggtitle("Average Rating by Since User's First Rating")




###############################################################
###                 Exploring genre effects                  ##

# Average rating by genre combination
genre_combos <- edx1 %>% group_by(genres) %>% summarize(avg = mean(rating), movies = n_distinct(movieId), n = n())

genre_combos %>% 
  ggplot(aes(genres, avg, size = n)) + ylim(0,5) +
  geom_point(alpha = 1/5, color = "blue")  + scale_x_discrete(name ="genres", labels = NULL) +
  ylab("ratings") + theme(legend.position = "none") + 
  ggtitle("Average Rating by Genre Combinations")

# Looking at number of genre combinations and ratings per combination
genre_combos %>% arrange(desc(movies))
range(genre_combos$movies)
range(genre_combos$n)
sum(genre_combos$movies < 5)
sum(genre_combos$n < 50)



###############################################################
##        Separating genres out  (duplicates movies)         ##

edx1 <- edx1 %>% relocate(genres, .after = last_col())
edx_g <- edx1 %>% separate_rows(genres, sep ="\\|") 


# Barcharts of average ratings & number of ratings for genre 
edx_g %>% group_by(genres) %>% summarise(avg = mean(rating), movies = n_distinct(movieId), n = n()) %>%
  ggplot(aes(x= avg, y = reorder(genres, avg), fill = -avg, label = n)) + xlim(0,5) +
  geom_bar(stat = "identity")  +  geom_text(size = 3, color = "#333333", hjust = 0, nudge_x = 0.03) +
  scale_fill_viridis() + xlab("Average Rating (with number of ratings)") + ylab("") + theme_bw(base_size = 9) + 
  theme(legend.position = "none", plot.title = element_text(size = 10), panel.grid = element_blank()) + 
  ggtitle("Average Rating by Genre")

edx_g %>% group_by(genres) %>% summarise(avg = mean(rating), movies = n_distinct(movieId), n = n()) %>%
  ggplot(aes(x= n, y = reorder(genres, n), fill = -n, label = sprintf("%0.2f", round(avg, digits = 2)))) +
  geom_bar(stat = "identity")  +  geom_text(size = 3, digits = 3,color = "#333333", hjust = 0, nudge_x = .03) +
  scale_fill_viridis() + xlab("Number of Ratings (with avg rating)") + ylab("") + theme_bw(base_size = 9) + 
  theme(legend.position = "none", plot.title = element_text(size = 10), panel.grid = element_blank()) + 
  ggtitle("Number of Ratings by Genre")

###--- Number of ratings vs avg rating by genre doesn't seem to follow same overall trend as number of rating/higher avg




# Looking into no genres listed ratings
edx1  %>% filter(genres == "(no genres listed)") %>% distinct(title)

# Exploring IMAX movies. Should IMAX be considered a genre?
IMAX <- edx1 %>% filter(str_detect(genres,"IMAX")) %>% group_by(movieId, title, genres) %>% summarize(avg = mean(rating), n = n())

IMAX %>% group_by(genres) %>% summarize(movies = n_distinct(movieId), observations = n(), average = mean(avg))
IMAX %>% arrange(desc(n))

n_distinct(IMAX$title)




##############################################################
##                  Updating bad data                       ##

#Updating no genre listed to drama  
edx1$genres[which(edx1$genres == "(no genres listed)")] <- "Drama"

#Removing IMAX from blockbuster movies which were shown on regular screens as well.
edx1 <-  edx1 %>% mutate(genres = ifelse(movieId %in% c(8965, 54001, 58559, 62999, 3159), str_remove(genres, "\\|IMAX"), genres))
edx1 %>% filter(str_detect(genres,"IMAX")) %>% summarize(avg = mean(rating), movies = n_distinct(movieId), n = n())

rm(IMAX, edx_g)


##############################################################
##         Pre-Processing: Adding age rated feature         ##

# Adding weeks since first rated variable to data
# Must convert timestamp from validation set & will parse out year released
validation1 <- validation %>% mutate(date_rated = as_datetime(timestamp), year = str_extract(title, "\\(\\d{4}\\)") %>% 
                                       str_remove_all("[\\(\\)]"), title = str_remove(title, "\\(\\d{4}\\)$") %>% str_trim())

#Removing hhmmss from date_rated & timestamp
validation1 <- validation1 %>% mutate(date_rated = date(date_rated)) %>% select(-timestamp)
#Calculating age of movie when rated & converting year to numeric
validation1 <- validation1 %>% mutate(age_rated = as.numeric(year(date_rated)) - as.numeric(year), year = as.numeric(year), week_rated = round_date(date_rated, unit = "week")) 

sum(validation1$age_rated < 0)
##- Found -1 age_rated values when movies rated just before year given in title. (critics?). Updated to 0.
validation1$age_rated[validation1$age_rated < 0] <- 0


#Discovering first date rated by user between sets and adding that date to both set, then calculating number of weeks since first rating
daysu_e <- edx1 %>% group_by(userId) %>% summarize(first_rated_e = min(date_rated))
daysu_v <- validation1 %>% group_by(userId) %>% summarize(first_rated_v = min(date_rated)) 
user_firsts <- merge(daysu_e, daysu_v, by="userId")
user_firsts <- user_firsts %>% mutate(first_rated = pmin(first_rated_e, first_rated_v)) %>% select(userId,first_rated)

edx1 <- merge(edx1, user_firsts, by = "userId", all = TRUE)  
edx1 <- edx1 %>% group_by(userId) %>% mutate(weeksrated_u = round((as.numeric(date_rated - first_rated))/7)) 







#################################################################
##       Pre-processing of validation set to match edx1        ##

# Updating no genres & IMAX blockbusters
validation1$genres[which(validation1$genres == "(no genres listed)")] <- "Drama"
validation1 <- validation1 %>% mutate(genres = ifelse(movieId %in% c(8965, 54001, 58559, 62999, 3159), str_remove(genres, "\\|IMAX"), genres))

# Adding weeks since user's first rating 
validation1 <- merge(validation1, user_firsts, by = "userId", all = TRUE)           
validation1 <- validation1 %>% group_by(userId) %>% mutate(weeksrated_u = round((as.numeric(date_rated - first_rated))/7))

# Removing columns not used in modeling from training and test sets
validation1 <- validation1 %>% select(-first_rated)
edx1 <- edx1 %>% select(-first_rated)






###############################################################
##     Splitting edx1 into train/test sets for modeling      ##

set.seed(1, sample.kind="Rounding")
index_edx <- createDataPartition(y = edx1$rating, times = 1, p = 0.1, list = FALSE)
train <- edx1[-index_edx,]
temp <- edx1[index_edx,]

test <- temp %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

removed <- anti_join(temp, test)

train <- rbind(train, removed)

rm(index_edx, temp, removed)


# Removing objects out of environment to increase available space.
rm(validation, edx1, user_firsts, daysu_v, daysu_e)


############################################################
##        Modeling ability of chosen predictors           ##

mu <- mean(train$rating) 

# Distribution of features for modeling

h_i <- train %>% group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu)) %>%
  ggplot(aes(b_i)) + geom_histogram(bins = 10) +
  theme_bw() + ggtitle("Movie Effect")

h_u <- train %>% group_by(userId) %>% 
  summarize(b_u = mean(rating - mu)) %>%
  ggplot(aes(b_u)) + geom_histogram(bins = 10) +
  theme_bw() + ggtitle("User Effect")

h_y <- train %>% group_by(year) %>% 
  summarize(b_y = mean(rating - mu)) %>%
  ggplot(aes(b_y)) + geom_histogram(bins = 10) +
  theme_bw() + ggtitle("Year Movie Released")

h_a <- train %>% group_by(age_rated) %>%
  summarize(b_a = mean(rating - mu)) %>%
  ggplot(aes(b_a)) + geom_histogram(bins = 10) + 
  theme_bw() + ggtitle("Age of Movie When Rated")

h_w <- train %>% group_by(week_rated) %>% 
  summarize(b_w = mean(rating - mu)) %>%
  ggplot(aes(b_w)) + geom_histogram(bins = 10) + 
  theme_bw() + ggtitle("Date Rated by Week")

h_uw <- train %>% group_by(weeksrated_u) %>%
  summarize(b_uw = mean(rating - mu)) %>%
  ggplot(aes(b_uw)) + geom_histogram(bins = 15) +
  theme_bw() + ggtitle("Weeks Since 1st Rating")

h_g <- train %>% group_by(genres) %>% 
  summarize(b_g = mean(rating - mu)) %>%
  ggplot(aes(b_g)) + geom_histogram(bins = 15) +
  theme_bw() + ggtitle("Genre Combinations")

gridExtra::grid.arrange(h_i, h_u, h_y, h_a, h_w, h_uw, h_g, ncol = 3)


rm(h_i, h_u, h_a, h_g, h_uw, h_w, h_y)






############################################################
####                  Modeling                          ####


## Base Model: Simplest recommendation, using the average of the data set 
mu <- mean(train$rating) 

just_avg <- RMSE(test$rating, mu)
rmse_results <- tibble(Model = "Naive Bayes", RMSE = just_avg)


## Model 1: modeling for movie effect
movie_avgs <- train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_ratings1 <- mu + test %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

model1_rmse <- RMSE(predicted_ratings1, test$rating)  

rmse_results <- bind_rows(rmse_results,
                         data_frame(Model = "Movie Effect", RMSE = model1_rmse ))
rmse_results 



## Model 2: movie & user-specific effects
user_avgs <- train %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings2 <- test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)


model2_rmse <- RMSE(predicted_ratings2, test$rating)

rmse_results <- bind_rows(rmse_results,
                         data_frame(Model = "Movie + User Effects", RMSE = model2_rmse))
rmse_results 


# checking prediction range
range(predicted_ratings2)

# Number of prediction below & above range
sum(predicted_ratings2 < .5)
sum(predicted_ratings2 > 5)

predicted_ratings2[predicted_ratings2 < 0.5] <- .5
predicted_ratings2[predicted_ratings2 > 5] <- 5

# Calculating RMSE with updated prediction limits
model2L_rmse <- RMSE(predicted_ratings2, test$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(Model = "Movie + User Effects with limits",  RMSE = model2L_rmse))
rmse_results 




## Regularization to minimize movies & users with few ratings over-impacting the model
# Using cross-validation to determine which value of lambda returns best prediction
lambdas <- seq(1, 6, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train$rating)
  
  b_i <- train %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>% 
    pull(pred)
  
  return(RMSE(predicted_ratings, test$rating))
})

qplot(lambdas, rmses)  
lambda <- lambdas[which.min(rmses)]
lambda 



## Model 3: Regularization for Movie & User Effects with lambda at 4.75
movieavgs_reg <- train %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda)) 

useravgs_reg <- train %>% 
  left_join(movieavgs_reg, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+lambda))

predicted_ratings_reg <- test %>% 
  left_join(movieavgs_reg, by='movieId') %>%
  left_join(useravgs_reg, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>% 
  pull(pred)

# Updating out-of-range predictions
predicted_ratings_reg[predicted_ratings_reg < 0.5] <- .5
predicted_ratings_reg[predicted_ratings_reg > 5] <- 5

# Calculating RMSE
model3_rmse <- RMSE(predicted_ratings_reg, test$rating)

rmse_results <- bind_rows(rmse_results,
                         data_frame(Model = "Movie + User Effects Regularized", RMSE = model3_rmse ))
rmse_results

# Removing large temp object
rm(predicted_ratings1, predicted_ratings2, predicted_ratings_reg)



## Model 4: including year movie was released
year_avgs <- train %>% 
  left_join(movieavgs_reg, by='movieId') %>%
  left_join(useravgs_reg, by='userId') %>%
  group_by(year) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u))

predicted_ratings4 <- test %>% 
  left_join(movieavgs_reg, by='movieId') %>%
  left_join(useravgs_reg, by='userId') %>%
  left_join(year_avgs, by='year') %>%
  mutate(pred = mu + b_i + b_u + b_y) %>% 
  pull(pred)

# Updating out-of-range predictions
predicted_ratings3[predicted_ratings4 < 0.5] <- .5
predicted_ratings3[predicted_ratings4 > 5] <- 5

# Calculating RMSE
model4_rmse <- RMSE(predicted_ratings4, test$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(Model = "Movie&User Regularized + Year Released Effects", RMSE = model4_rmse))
rmse_results 



## Model 5: with age of movie when rated 
age_rated_avgs <- train %>% 
  left_join(movieavgs_reg, by='movieId') %>%
  left_join(useravgs_reg, by='userId') %>%
  left_join(year_avgs, by='year') %>%
  group_by(age_rated) %>%
  summarize(b_a = mean(rating - mu - b_i - b_u - b_y))

predicted_ratings5 <- test %>% 
  left_join(movieavgs_reg, by='movieId') %>%
  left_join(useravgs_reg, by='userId') %>%
  left_join(year_avgs, by='year') %>%
  left_join(age_rated_avgs, by='age_rated') %>%
  mutate(pred = mu + b_i + b_u + b_y + b_a) %>% 
  pull(pred)

# Updating out-of-range predictions
predicted_ratings5[predicted_ratings5 < 0.5] <- .5
predicted_ratings5[predicted_ratings5 > 5] <- 5

# Calculating RMSE
model5_rmse <- RMSE(predicted_ratings5, test$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(Model= "Movie/User Regularized + Year + Age When Rated", RMSE = model5_rmse ))
rmse_results 



## Model 6: with date movie was rated (rounded to week)
week_rated_avgs <- train %>% 
  left_join(movieavgs_reg, by='movieId') %>%
  left_join(useravgs_reg, by='userId') %>%
  left_join(year_avgs, by='year') %>%
  left_join(age_rated_avgs, by='age_rated') %>%
  group_by(week_rated) %>%
  summarize(b_d = mean(rating - mu - b_i - b_u - b_y - b_a))

predicted_ratings6 <- test %>% 
  left_join(movieavgs_reg, by='movieId') %>%
  left_join(useravgs_reg, by='userId') %>%
  left_join(year_avgs, by='year') %>%
  left_join(age_rated_avgs, by='age_rated') %>%
  left_join(week_rated_avgs, by='week_rated') %>%
  mutate(pred = mu + b_i + b_u + b_y + b_a + b_d) %>% 
  pull(pred)

# Updating out-of-range predictions
predicted_ratings6[predicted_ratings6 < 0.5] <- .5
predicted_ratings6[predicted_ratings6 > 5] <- 5

# Calculating RMSE
model6_rmse <- RMSE(predicted_ratings6, test$rating)

rmse_results <- bind_rows(rmse_results,
                         data_frame(Model="Movie&User Regularized + Year + Age & Week When Rated", RMSE = model6_rmse ))
rmse_results 


# Removing large temp objects
rm(predicted_ratings4, predicted_ratings5, predicted_ratings6)



# Model 7: with number of weeks since user's 1st rating
weeksratedu_avgs <- train %>% 
  left_join(movieavgs_reg, by='movieId') %>%
  left_join(useravgs_reg, by='userId') %>% 
  left_join(year_avgs, by='year') %>%
  left_join(age_rated_avgs, by='age_rated') %>%
  left_join(week_rated_avgs, by='week_rated') %>%
  group_by(weeksrated_u) %>%
  summarize(b_uw = mean(rating - mu - b_i - b_u - b_y - b_a - b_d))

predicted_ratings7 <- test %>% 
  left_join(movieavgs_reg, by='movieId') %>%
  left_join(useravgs_reg, by='userId') %>%
  left_join(year_avgs, by='year') %>%
  left_join(age_rated_avgs, by='age_rated') %>%
  left_join(week_rated_avgs, by='week_rated') %>%
  left_join(weeksratedu_avgs, by='weeksrated_u') %>%
  mutate(pred = mu + b_i + b_u + b_y + b_a + b_d + b_uw) %>% 
  pull(pred)

# Updating out-of-range predictions
predicted_ratings7[predicted_ratings7 < 0.5] <- .5
predicted_ratings7[predicted_ratings7 > 5] <- 5

# Calculating RMSE
model7_rmse <- RMSE(predicted_ratings7, test$rating)

rmse_results <- bind_rows(rmse_results,
                         data_frame(Model="Movie&User Reg + Year + Age&Week Rated + User Rating Weeks", RMSE = model7_rmse))
rmse_results

# Removing large temp object
rm(predicted_ratings7)



## Model 8: with combinations of genres effect
genre_avgs <- train %>% 
  left_join(movieavgs_reg, by='movieId') %>%
  left_join(useravgs_reg, by='userId') %>%
  left_join(year_avgs, by='year') %>%
  left_join(age_rated_avgs, by='age_rated') %>%
  left_join(week_rated_avgs, by='week_rated') %>%
  left_join(weeksratedu_avgs, by='weeksrated_u') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u - b_y - b_a - b_d - b_uw)) 

predicted_ratings8 <- test %>% 
  left_join(movieavgs_reg, by='movieId') %>%
  left_join(useravgs_reg, by='userId') %>%
  left_join(year_avgs, by='year') %>%
  left_join(age_rated_avgs, by='age_rated') %>%
  left_join(week_rated_avgs, by='week_rated') %>%
  left_join(weeksratedu_avgs, by='weeksrated_u') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_y + b_a + b_d + b_uw + b_g) %>% 
  pull(pred)

# Updating out-of-range predictions
predicted_ratings8[predicted_ratings8 < 0.5] <- .5
predicted_ratings8[predicted_ratings8 > 5] <- 5

# Calculating RMSE
model8rmse <- RMSE(predicted_ratings8, test$rating)

rmse_results <- bind_rows(rmse_results,
                         data_frame(Model="Movie&User Reg + Year + Age&Week Rated + UserWeeks + GenreCombos",  RMSE = model8_rmse))
rmse_results 



## Model 9: Addition to effects model 8: Matrix Factorization

# Creating residuals for matrix 
residuals <- train %>% 
  left_join(movieavgs_reg, by='movieId') %>%
  left_join(useravgs_reg, by='userId') %>%
  left_join(year_avgs, by='year') %>%
  left_join(age_rated_avgs, by='age_rated') %>%
  left_join(week_rated_avgs, by='week_rated') %>%
  left_join(weeksratedu_avgs, by='weeksrated_u') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(b_res = (rating - mu - b_i - b_u - b_y - b_a - b_d - b_uw - b_g)) %>% select(userId, movieId, b_res)

test_residuals <- test %>% 
  left_join(movieavgs_reg, by='movieId') %>%
  left_join(useravgs_reg, by='userId') %>%
  left_join(year_avgs, by='year') %>%
  left_join(age_rated_avgs, by='age_rated') %>%
  left_join(week_rated_avgs, by='week_rated') %>%
  left_join(weeksratedu_avgs, by='weeksrated_u') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(b_res = (rating - mu - b_i - b_u - b_y - b_a - b_d - b_uw - b_g)) %>% select(userId, movieId, b_res)

# removing large objects from environments
rm(predicted_ratings7)



# transforming training & test residuals to matrix
residuals <- residuals %>% as.matrix()
test_residuals <- test_residuals %>% as.matrix()

# removing train to make room for large residuals
rm(train)




######################
# Matrix Factorization with Recosystem

# write residual and test matrices on disk & used # use Recosystem option data_file() to specify source of dataset.
write.table(residuals, file = "trainset.txt" , sep = " " , row.names = FALSE, col.names = FALSE)
write.table(test_residuals, file = "testset.txt" , sep = " " , row.names = FALSE, col.names = FALSE)

# Increasing memory limit
memory.limit(size=48000)

set.seed(123) # This is a randomized algorithm
train_set = data_file("trainset.txt")
test_set = data_file("testset.txt")

# build a recommender object
r <-Reco()

# tuning training set. using CRAN.R-project tuning parameters.  This takes a long time!
opts <- r$tune(train_set, opts = list(dim = c(10, 20, 30), lrate = c(0.1, 0.2),
                                      costp_l1 = 0, costq_l1 = 0,
                                      nthread = 1, niter = 10))
opts
opts$min

# training recosystem model
r$train(train_set, opts = c(opts$min, nthread = 1, niter = 20)) 

# predicting residuals
pred_file <- tempfile()
r$predict(test_set, out_file(pred_file))  
predicted_residuals <- scan(pred_file)


# Model 8 predicted ratings + matrix factorization residuals
predicted_ratings_mf <- predicted_ratings8 + predicted_residuals

range(predicted_ratings_mf)

# Updating out-of-range predictions
predicted_ratings_mf[predicted_ratings_mf < 0.5] <- .5
predicted_ratings_mf[predicted_ratings_mf > 5] <- 5

model_mf_rmse <- RMSE(predicted_ratings_mf,test$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="Movie&User Reg + Year + Age&WeekRated + UserWeeks + Genres + Matrix Factorization",  
                                     RMSE = model_mf_rmse))
rmse_results


rm(predicted_residuals, predicted_ratings_mf)
rm(test_residuals, residuals)

       

#####################################################
##                Final Validation                 ##


# Calculating predictions for final validation using Model 8
predicted_ratingsv <- validation1 %>% 
  left_join(movieavgs_reg, by='movieId') %>%
  left_join(useravgs_reg, by='userId') %>%
  left_join(year_avgs, by='year') %>%
  left_join(age_rated_avgs, by='age_rated') %>%
  left_join(week_rated_avgs, by='week_rated') %>%
  left_join(weeksratedu_avgs, by='weeksrated_u') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_y + b_a + b_d + b_uw + b_g) %>% 
  pull(pred)


# checking prediction range
range(predicted_ratingsv)
# Updating out-of-range predictions
predicted_ratingsv[predicted_ratingsv < 0.5] <- .5
predicted_ratingsv[predicted_ratingsv > 5] <- 5

# Calculating RMSE
final8_rmse <- RMSE(predicted_ratingsv, validation1$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="Final: Movie&User Reg + Year + Age&WeekRated + UserWeeks + Genres",  RMSE = final8_rmse))
rmse_results 

# Removing large objects
rm(predicted_ratings8,predicted_residuals,residuals, test_residuals)


## 
# Calculating validation residuals for matrix factorization
valid_residuals <- validation1 %>% 
  left_join(movieavgs_reg, by='movieId') %>%
  left_join(useravgs_reg, by='userId') %>%
  left_join(year_avgs, by='year') %>%
  left_join(age_rated_avgs, by='age_rated') %>%
  left_join(week_rated_avgs, by='week_rated') %>%
  left_join(weeksratedu_avgs, by='weeksrated_u') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(b_res = (rating - mu - b_i - b_u - b_y - b_a - b_d - b_uw - b_g)) %>% select(userId, movieId, b_res)

valid_residuals <- valid_residuals %>% as.matrix()

# write validation matrix on disk & used # use recosystem option data_file() to specify source of dataset.
write.table(valid_residuals, file = "validset.txt" , sep = " " , row.names = FALSE, col.names = FALSE)

set.seed(123) # This is a randomized algorithm
valid_set = data_file("validset.txt")


# predicting residuals
pred_file_v <- tempfile()
r$predict(valid_set, out_file(pred_file_v))  
valid_residualsv <- scan(pred_file_v)

# predicting ratings as model 8 predictions + recosystem predicted residuals
predicted_ratings_final <- predicted_ratingsv + predicted_residualsv

range(predicted_ratings_final)
# Updating out-of-range predictions
predicted_ratings_final[predicted_ratings_final < 0.5] <- .5
predicted_ratings_final[predicted_ratings_final > 5] <- 5

final_mf_rmse <- RMSE(predicted_ratings_final,validation1$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="Final: Movie&User Reg + Year + Age&WeekRated + UserWeeks + Gernres + MatrixFactorzn",  
                                     RMSE = final_mf_rmse ))



# Final predictions Comparison 
final_pred <- as.data.frame(predicted_ratings_final)
valid2 <- cbind(validation1, final_pred)

# Most rated movies
top10 <- valid2 %>% group_by(title) %>% summarize(n = n(), avg = mean(rating), prediction= mean(predicted_ratings_final)) %>% arrange(desc(n)) %>% slice(1:10)
top10

# Movies w n 100:1000
top10_avg3 <- valid2 %>% group_by(title) %>% summarize(n = n(), avg = mean(rating), prediction= mean(predicted_ratings_final)) %>% 
                    arrange(desc(avg)) %>% filter(n > 100 & n < 1000 & avg < 3) %>% slice(1:10) 
top10_avg3

# movies w n 50:100
top10_3 <- valid2 %>% group_by(title) %>% summarize(n = n(), avg = mean(rating), prediction= mean(predicted_ratings_final)) %>% 
                    arrange(desc(avg)) %>% filter(n > 50 & n < 100) %>% slice(1:10) 
top10_3


top10_avg3 <- top10_avg3 %>% mutate(title = sub("\\(.*)", "", title))


file <- "yelp_training_set_business.json"
conn <- file(file, "r")
input <- readLines(conn, -1L)
test <- lapply(input, fromJSON)
test <- lapply(test, cbind)
test <- as.data.frame(test)
business_set <- as.data.frame(t(test))
row.names(business_set) <- seq(1, nrow(business_set))

file <- "yelp_training_set_user.json"
conn <- file(file, "r")
input <- readLines(conn, -1L)
test <- lapply(input, fromJSON)
test <- lapply(test, cbind)
test <- as.data.frame(test)
user_set <- as.data.frame(t(test))
row.names(user_set) <- seq(1, nrow(user_set))

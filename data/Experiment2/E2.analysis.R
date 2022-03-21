#this is for E2 from the verb-verb series

#pull in files

#first, pull in all the ixs files
ff = read.table(file.choose(), header = T, sep = ",")
fp = read.table(file.choose(), header = T, sep = ",")
gp = read.table(file.choose(), header = T, sep = ",")
tt = read.table(file.choose(), header = T, sep = ",")
ro = read.table(file.choose(), header = T, sep = ",")

#name the columns to be unique in each case
colnames(ff) = c("seq", "subj", "item", "cond",
                 "ffR1", "ffR2", "ffR3", "ffR4", "ffR5", "ffR6", "ffR7", "ffR8")
colnames(fp) = c("seq", "subj", "item", "cond",
                 "fpR1", "fpR2", "fpR3", "fpR4", "fpR5", "fpR6", "fpR7", "fpR8")
colnames(gp) = c("seq", "subj", "item", "cond",
                 "gpR1", "gpR2", "gpR3", "gpR4", "gpR5", "gpR6", "gpR7", "gpR8")
colnames(tt) = c("seq", "subj", "item", "cond",
                 "ttR1", "ttR2", "ttR3", "ttR4", "ttR5", "ttR6", "ttR7", "ttR8")
colnames(ro) = c("seq", "subj", "item", "cond",
                 "roR1", "roR2", "roR3", "roR4", "roR5", "roR6", "roR7", "roR8")

#remove all the 0 conditions from all files

ff = ff[ff$cond > 0,]
fp = fp[fp$cond > 0,]
gp = gp[gp$cond > 0,]
tt = tt[tt$cond > 0,]
ro = ro[ro$cond > 0,]

#1858 rows in all cases

#successive merge (I'm sure there's a more efficient way
#to do this)
m1 = merge(ff, fp, by = (c("seq", "subj", "item", "cond")))
m2 = merge(m1, gp)
m3 = merge(m2,tt)
m4 = merge(m3,ro)

#good.  replace all NAs with 0

m4[is.na(m4)] = 0

#great.  rename.

d = m4

#recode condition
#1 is short ORC
#2 is short ORC with phrasal verb
#3 is long ORC
#4 is long ORC phrasal
#5 is short SRC
#6 is long SRC phrasal
d$rc = ifelse(d$cond > 4, "SRC", 
              ifelse(d$cond ==1 | d$cond ==3, 
                     "ORC", "ORCphrasal"))
d$length = ifelse(d$cond == 1|d$cond == 2|d$cond == 5,
                  "short", "long")

#contrast coding

d$length.cent = ifelse(d$length == "short", -.5, .5)

d$rc = as.factor(d$rc)
contrasts(d$rc)

contrasts(d$rc) = cbind(c(.5, .5, -.5), c(-.5, .5, 0))
#try contrasts suggested by Brian
contrasts(d$rc) = cbind(c(.25, .25, -.5), c(-.5, .5, 0))

> contrasts(d$rc)
[,1] [,2]
ORC         0.5 -0.5
ORCphrasal  0.5  0.5
SRC        -0.5  0.0

#make subject and item into factors

d$subj = as.factor(d$subj)
d$item = as.factor(d$item)

#embedded NP is region 3 for the ORCs, region 4 for the SRCs
d$ff.NP = ifelse(d$rc == "SRC", d$ffR4, d$ffR3)
d$fp.NP = ifelse(d$rc == "SRC", d$fpR4, d$fpR3)
d$gp.NP = ifelse(d$rc == "SRC", d$gpR4, d$gpR3)
d$tt.NP = ifelse(d$rc == "SRC", d$ttR4, d$ttR3)
d$ro.NP = ifelse(d$rc == "SRC", d$roR4, d$roR3)

#embedded verb is region 4 for the ORCs, region 3 for the SRCs
d$ff.verb = ifelse(d$rc == "SRC", d$ffR3, d$ffR4)
d$fp.verb = ifelse(d$rc == "SRC", d$fpR3, d$fpR4)
d$gp.verb = ifelse(d$rc == "SRC", d$gpR3, d$gpR4)
d$tt.verb = ifelse(d$rc == "SRC", d$ttR3, d$ttR4)
d$ro.verb = ifelse(d$rc == "SRC", d$roR3, d$roR4)

#spit out file.

setwd("/Users/Adrian/Desktop/forCaroline/")
write.csv(d, file = "E2.alldata.csv")

#restrict to trials where there was a first fix on the NP, on the
#embedded verb, or on the matrix verb

d.NP = d[d$ff.NP > 0,]
d.verb = d[d$ff.verb > 0,]
d.matrix = d[d$ffR6 > 0,]

#and for intervening material, in long conditions
d.inter = d[d$ffR5 > 0,]

library(sciplot)

setwd("/Users/Adrian/Dropbox/Research/CurrentProjects/Rel Clause Again/E2_RClength")
png("eyetracking_means_2.png", width = 12, height = 12, units = 'in', res = 600)

par(mfrow = c(4, 4))

#NP
bargraph.CI(data = d.NP, x.factor = rc, 
            response = ff.NP, group = length, 
            err.width = 0,
            legend = T, leg.lab = c("long", "short"), 
            ylim = c(100, 300), ylab = "ms",
            xlab = "RC Type", 
            main = "first fix, RC NP")

bargraph.CI(data = d.NP, x.factor = rc, 
            response = fp.NP, group = length, 
            err.width = 0,
            #legend = T, leg.lab = c("long", "short"), 
            ylim = c(300, 500), ylab = "ms",
            xlab = "RC Type", 
            main = "first pass, RC NP")

bargraph.CI(data = d.NP, x.factor = rc, 
            response = gp.NP, group = length, 
            err.width = 0,
            #legend = T, leg.lab = c("long", "short"), 
            ylim = c(400, 600), ylab = "ms",
            xlab = "RC Type", 
            main = "go-past, RC NP")

bargraph.CI(data = d.NP, x.factor = rc, 
            response = ro.NP, group = length, 
            err.width = 0,
            #legend = T, leg.lab = c("long", "short"), 
            lc = F, uc = F,
            ylim = c(0, .5), ylab = "proportion",
            xlab = "RC Type", 
            main = "regressions, RC NP")

#RC verb
bargraph.CI(data = d.verb, x.factor = rc, 
            response = ff.verb, group = length, 
            err.width = 0,
            legend = T, leg.lab = c("long", "short"), 
            ylim = c(100, 300), ylab = "ms",
            xlab = "RC Type", 
            main = "first fix, RC verb")

bargraph.CI(data = d.verb, x.factor = rc, 
            response = fp.verb, group = length, 
            err.width = 0,
            #legend = T, leg.lab = c("long", "short"), 
            ylim = c(200, 400), ylab = "ms",
            xlab = "RC Type", 
            main = "first pass, RC verb")

bargraph.CI(data = d.verb, x.factor = rc, 
            response = gp.verb, group = length, 
            err.width = 0,
            #legend = T, leg.lab = c("long", "short"), 
            ylim = c(320, 520), ylab = "ms",
            xlab = "RC Type", 
            main = "go-past, RC verb")

bargraph.CI(data = d.verb, x.factor = rc, 
            response = ro.verb, group = length, 
            err.width = 0,
            #legend = T, leg.lab = c("long", "short"), 
            lc = F, uc = F,
            ylim = c(0, .5), ylab = "proportion",
            xlab = "RC Type", 
            main = "regressions, RC verb")

#intervening material

bargraph.CI(data = d.inter, x.factor = rc, 
            response = ffR5, 
            #group = length, 
            err.width = 0,
            #legend = T, leg.lab = c("long", "short"), 
            ylim = c(100, 300), ylab = "ms",
            xlab = "RC Type", 
            space = 1,
            col = gray(.3),
            main = "first fix, prep phrase")

bargraph.CI(data = d.inter, x.factor = rc, 
            response = fpR5, 
            #group = length, 
            err.width = 0,
            #legend = T, leg.lab = c("long", "short"), 
            ylim = c(300, 500), ylab = "ms",
            xlab = "RC Type", 
            space = 1,
            col = gray(.3),
            main = "first pass, prep phrase")

bargraph.CI(data = d.inter, x.factor = rc, 
            response = gpR5, 
            #group = length, 
            err.width = 0,
            #legend = T, leg.lab = c("long", "short"), 
            ylim = c(400, 600), ylab = "ms",
            xlab = "RC Type", 
            space = 1,
            col = gray(.3),
            main = "go-past, prep phrase")


bargraph.CI(data = d.inter, x.factor = rc, 
            response = roR5, 
            #group = length, 
            err.width = 0,
            #legend = T, leg.lab = c("long", "short"), 
            lc = F, uc = F,
            ylim = c(0,.5), ylab = "ms",
            xlab = "RC Type", 
            space = 1,
            col = gray(.3),
            main = "regressions, prep phrase")

#matrix
bargraph.CI(data = d.matrix, x.factor = rc, 
            response = ffR6, group = length, 
            err.width = 0,
            legend = T, leg.lab = c("long", "short"),
            x.leg = 6, y.leg = 310,
            ylim = c(100, 300), ylab = "ms",
            xlab = "RC Type", 
            main = "first fix, matrix verb")

bargraph.CI(data = d.matrix, x.factor = rc, 
            response = fpR6, group = length, 
            err.width = 0,
            #legend = T, leg.lab = c("long", "short"), 
            ylim = c(200, 400), ylab = "ms",
            xlab = "RC Type", 
            main = "first pass, matrix verb")

bargraph.CI(data = d.matrix, x.factor = rc, 
            response = gpR6, group = length, 
            err.width = 0,
            #legend = T, leg.lab = c("long", "short"), 
            ylim = c(300, 500), ylab = "ms",
            xlab = "RC Type", 
            main = "go-past, matrix verb")

bargraph.CI(data = d.matrix, x.factor = rc, 
            response = roR6, group = length, 
            err.width = 0,
            #legend = T, leg.lab = c("long", "short"), 
            lc = F, uc = F,
            ylim = c(0, .5), ylab = "proportion",
            xlab = "RC Type", 
            main = "regressions, matrix verb")

dev.off()


#for rc NP
model.ff.NP = lmer(ff.NP~rc * length.cent + (1+rc+length.cent|subj) +
                     (1+rc+length.cent|item), data = d.NP)


summary(model.ff.NP)

#needed to remove random item slopes to converge
model.fp.NP = lmer(fp.NP~rc * length.cent + (1+rc+length.cent|subj) +
                     (1|item), data = d.NP)

summary(model.fp.NP)

model.gp.NP = lmer(gp.NP~rc * length.cent + (1+rc+length.cent|subj) +
                     (1+rc+length.cent|item), data = d.NP)

summary(model.gp.NP)

#needed to remove random item slopes
model.ro.NP = glmer(ro.NP~rc * length.cent + (1+rc + length.cent|subj) +
                      (1|item), data = d.NP, family = "binomial")

summary(model.ro.NP)

#for verb
#need to leave out phrasal conditions
d.nophrase = d.verb[d.verb$rc != "ORCphrasal",]
d.nophrase$rc.cent = d.nophrase$rc.cent = ifelse(d.nophrase$rc == "SRC", -.5, .5)

model.ff.verb = lmer(ff.verb~rc.cent * length.cent + (1+rc+length.cent|subj) +
                     (1+rc+length.cent|item), data = d.nophrase)


summary(model.ff.verb)

model.fp.verb = lmer(fp.verb~rc.cent * length.cent + (1+rc+length.cent|subj) +
                       (1+rc+length.cent|item), data = d.nophrase)


summary(model.fp.verb)

model.gp.verb = lmer(gp.verb~rc.cent * length.cent + (1+rc+length.cent|subj) +
                       (1+rc+length.cent|item), data = d.nophrase)


summary(model.gp.verb)

model.ro.verb = glmer(ro.verb~rc.cent * length.cent + (1+rc+length.cent|subj) +
                       (1|item), data = d.nophrase, family = "binomial")


summary(model.ro.verb)


#and prep phrase

model.ff.inter = lmer(ffR5~rc + (1+rc|subj) +
                        (1+rc|item), data = d.inter)

summary(model.ff.inter)

model.fp.inter = lmer(fpR5~rc + (1+rc|subj) +
                        (1+rc|item), data = d.inter)

summary(model.fp.inter)

model.gp.inter = lmer(gpR5~rc + (1+rc|subj) +
                        (1+rc|item), data = d.inter)

summary(model.gp.inter)

model.ro.inter = glmer(roR5~rc + (1+rc|subj) +
                        (1+rc|item), data = d.inter, family = "binomial")

summary(model.ro.inter)

#matrix verb

model.ff.matrix = lmer(ffR6~rc * length.cent + (1+rc+length.cent|subj) +
                     (1+rc+length.cent|item), data = d.matrix)


summary(model.ff.matrix)

model.fp.matrix = lmer(fpR6~rc * length.cent + (1+rc+length.cent|subj) +
                         (1+rc+length.cent|item), data = d.matrix)


summary(model.fp.matrix)

model.gp.matrix = lmer(gpR6~rc * length.cent + (1+rc+length.cent|subj) +
                         (1+rc+length.cent|item), data = d.matrix)


summary(model.gp.matrix)

model.ro.matrix = glmer(roR6~rc * length.cent + (1+rc+length.cent|subj) +
                         (1|item), data = d.matrix, family = "binomial")


summary(model.ro.matrix)

#extra analysis to compare each condition to ORC short

d.nophrase = d.matrix[d.matrix$rc != "ORCphrasal",]

d.nophrase$cond = as.factor(d.nophrase$cond)

model.ff.matrix.new = lmer(ffR6~cond + (1|subj) + 
                             (1|item), data = d.nophrase)

summary(model.ff.matrix.new)

model.fp.matrix.new = lmer(fpR6~cond + (1|subj) + 
                             (1|item), data = d.nophrase)

summary(model.fp.matrix.new)

model.gp.matrix.new = lmer(gpR6~cond + (1|subj) + 
                             (1|item), data = d.nophrase)

summary(model.gp.matrix.new)

#let's do a combined analysis with the same four conditions from the
#other experiment (Experiment 1 in paper; E4 in sequence)

head(d.nophrase)
#for simplicity, get just critical columns for matrix verb

d2 = d.nophrase[,c(2,3,4,10,18,26,42,45,46)]

#matrix data from other experiment
head(d.matrix.4)
d4 = d.matrix.4[,c(2,3,4,10,18,26,42,45,46)]
head(d4)

d2$exp = rep(2,nrow(d2))
d4$exp = rep(4,nrow(d4))

#combine them
d4$cond = as.factor(d4$cond)
d.comb = rbind(d2, d4)

#now make the model
#make a new column that codes the four conditions
d.comb$newcond = ifelse(d.comb$rc == "ORC" & d.comb$length == "short",
                        "a", ifelse(d.comb$rc == "ORC" & d.comb$length == "long","b",
                                    ifelse(d.comb$length == "short","c","d")))
#a is orc short
#b is orc long
#c is src short
#d is src long

#code exp as a factor

d.comb$exp = as.factor(d.comb$exp)

#oops, need to center

d.comb$exp.cent = ifelse(d.comb$exp == 4, -.5, .5)

#make a model

#to eliminate common subject numbers, add 100 to subject number from 
#second experiment

head(d.comb)
d.comb$subj = as.numeric(as.character(d.comb$subj))
#now add 100
d.comb$subj = ifelse(d.comb$exp.cent == .5, d.comb$subj + 100, d.comb$subj)

#now go back to factor
d.comb$subj = as.factor(d.comb$subj)

comb.ff.model = lmer(ffR6 ~ newcond*exp.cent + 
                       (1|subj) + 
                       (1|item), data = d.comb, REML = FALSE)

summary(comb.ff.model)

Fixed effects:
  Estimate Std. Error t value
(Intercept)        259.539      4.924   52.71
newcondb           -30.432      4.615   -6.59
newcondc           -12.203      4.581   -2.66
newcondd           -15.903      4.580   -3.47
exp.cent            -4.059      9.160   -0.44
newcondb:exp.cent    6.891      9.223    0.75
newcondc:exp.cent   15.146      9.162    1.65
newcondd:exp.cent    5.026      9.159    0.55

#now add random slopes

comb.ff.model.2 = lmer(ffR6 ~ newcond*exp.cent + 
                         (1+newcond|subj) + 
                         (1+newcond|item), data = d.comb, REML = FALSE)

summary(comb.ff.model.2)

#yes, that works!  But random effects are extremely highly correlated.
#remove random slopes.

#and random slopes to model without exp.cent

comb.ff.model.3 = lmer(ffR6 ~ newcond + 
                         (1+newcond|subj) + 
                         (1+newcond|item), data = d.comb, REML = FALSE)

summary(comb.ff.model.3)

#does including those slopes help?




comb.fp.model.3 = lmer(fpR6 ~ newcond + 
                       (1+newcond|subj) + 
                       (1+newcond|item), data = d.comb, REML = FALSE)

summary(comb.fp.model.3)

#what if we get rid of correlations?

comb.fp.model.4 = lmer(fpR6 ~ newcond + 
                         (1|subj) + (0 + newcondb|subj) +
                         (0 + newcondc|subj) + (0+ newcondd|subj) +
                         (1|item), data = d.comb, REML = FALSE)

summary(comb.fp.model.4)

comb.fp.model.4 = lmer(fpR6 ~ newcond + 
                         (1+newcond|exp.cent/subj) + 
                         (1+newcond|item), data = d.comb, REML = FALSE)

summary(comb.fp.model.4)

anova(comb.fp.model.3, comb.fp.model.4)

comb.gp.model.3 = lmer(gpR6 ~ newcond*exp.cent + 
                         (1+newcond|exp.cent/subj) + 
                         (1+newcond|item), data = d.comb, REML = FALSE)

summary(comb.gp.model.3)

#remove only subject random slopes?

comb.gp.model.xx = lmer(gpR6 ~ newcond*exp.cent + 
                         (1|subj) + 
                         (1+newcond|item), data = d.comb, REML = FALSE)

summary(comb.gp.model.xx)

comb.gp.model.4 = lmer(gpR6 ~ newcond + 
                         (1+newcond|exp.cent/subj) + 
                         (1+newcond|item), data = d.comb, REML = FALSE)

summary(comb.gp.model.4)


anova(comb.gp.model.3, comb.gp.model.4)




comb.ro.model.3 = glmer(roR6 ~ newcond*exp.cent + 
                       (1 + newcond|exp.cent/subj) + 
                       (1|item), data = d.comb, family = "binomial", REML = FALSE)

summary(comb.ro.model.3)

comb.ro.model.4 = glmer(roR6 ~ newcond*exp.cent + 
                          (1|exp.cent/subj) + 
                          (1|item), data = d.comb, family = "binomial", REML = FALSE)

summary(comb.ro.model.4)

#ack!  what a mess this all is.  what if we start over without random slopes?

comb.ff.model = lmer(ffR6 ~ newcond*exp.cent + 
                       (1|exp.cent/subj) + 
                       (1|item), data = d.comb, REML = FALSE)

summary(comb.ff.model)

comb.fp.model = lmer(fpR6 ~ newcond*exp.cent + 
                       (1|exp.cent/subj) + 
                       (1|item), data = d.comb, REML = FALSE)

summary(comb.fp.model)

comb.gp.model = lmer(gpR6 ~ newcond*exp.cent + 
                       (1|exp.cent/subj) + 
                       (1|item), data = d.comb, REML = FALSE)

summary(comb.gp.model)

comb.gp.model.x = lmer(gpR6 ~ newcond*exp.cent + 
                       (1|exp.cent/subj) + 
                       (1|item), data = d.comb, REML = FALSE)

summary(comb.gp.model.x)


comb.ro.model = glmer(roR6 ~ newcond*exp.cent + 
                       (1|exp.cent/subj) + 
                       (1|item), data = d.comb, REML = FALSE, family = "binomial")

summary(comb.ro.model)


d.comb = droplevels(d.comb)

setwd("/Users/Adrian/Dropbox/Research/CurrentProjects/Rel Clause Again")
png("Figure3x.png", width = 12, height = 3, units = 'in', res = 600)

par(mfrow = c(1, 4))

bargraph.CI(data = d.comb, x.factor = rc, 
            response = ffR6, group = length, 
            err.width = 0,
            legend = T, leg.lab = c("long", "short"),
            x.leg = 4, y.leg = 310,
            ylim = c(100, 300), ylab = "ms",
            xlab = "RC Type", 
            main = "first fix, matrix verb")

bargraph.CI(data = d.comb, x.factor = rc, 
            response = fpR6, group = length, 
            err.width = 0,
            #legend = T, leg.lab = c("long", "short"),
            #x.leg = 6, y.leg = 310,
            ylim = c(200, 400), ylab = "ms",
            xlab = "RC Type", 
            main = "first pass, matrix verb")

bargraph.CI(data = d.comb, x.factor = rc, 
            response = gpR6, group = length, 
            err.width = 0,
            #legend = T, leg.lab = c("long", "short"),
            #x.leg = 6, y.leg = 310,
            ylim = c(300, 500), ylab = "ms",
            xlab = "RC Type", 
            main = "go-past, matrix verb")

bargraph.CI(data = d.comb, x.factor = rc, 
            response = roR6, group = length, 
            err.width = 0,
            #legend = T, leg.lab = c("long", "short"),
            #x.leg = 6, y.leg = 310,
            ylim = c(0,.5), ylab = "ms",
            lc = F, uc = F,
            xlab = "RC Type", 
            main = "regressions out, matrix verb")

dev.off()


#sanity check.  do anovas on go-past at matrix verb.

head(d.comb)

ezANOVA(data = d.comb, dv = ffR6, within = .(rc, length), wid = item)

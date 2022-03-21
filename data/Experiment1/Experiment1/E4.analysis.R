#this is for E4 from the verb-verb series

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

#2197 rows in all cases

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
d$rc = ifelse(d$cond > 12, "SRC", "ORC")
d$length = ifelse(d$cond == 11|d$cond == 13, "short", "long")

#embedded NP is region 3 for the ORCs, region 4 for the SRCs
d$ff.NP = ifelse(d$rc == "ORC", d$ffR3, d$ffR4)
d$fp.NP = ifelse(d$rc == "ORC", d$fpR3, d$fpR4)
d$gp.NP = ifelse(d$rc == "ORC", d$gpR3, d$gpR4)
d$tt.NP = ifelse(d$rc == "ORC", d$ttR3, d$ttR4)
d$ro.NP = ifelse(d$rc == "ORC", d$roR3, d$roR4)

#embedded verb is region 4 for the ORCs, region 3 for the SRCs
d$ff.verb = ifelse(d$rc == "ORC", d$ffR4, d$ffR3)
d$fp.verb = ifelse(d$rc == "ORC", d$fpR4, d$fpR3)
d$gp.verb = ifelse(d$rc == "ORC", d$gpR4, d$gpR3)
d$tt.verb = ifelse(d$rc == "ORC", d$ttR4, d$ttR3)
d$ro.verb = ifelse(d$rc == "ORC", d$roR4, d$roR3)

#center the rc and length variables

d$rc.cent = ifelse(d$rc == "SRC", -.5, .5)
d$length.cent = ifelse(d$length == "short", -.5, .5)

#make subject and item into factors

d$subj = as.factor(d$subj)
d$item = as.factor(d$item)

#restrict to trials where there was a first fix on the NP, on the
#embedded verb, or on the matrix verb

d.NP = d[d$ff.NP > 0,]
d.verb = d[d$ff.verb > 0,]
d.matrix = d[d$ffR6 > 0,]

#and for intervening material, in long conditions
d.inter = d[d$ffR5 > 0,]
  
library(sciplot)

#let's look at first fix, first pass, go-past, and regressions
#for each of the three critical regions, for each condition


setwd("/Users/Adrian/Dropbox/Research/CurrentProjects/Rel Clause Again")
png("eyetracking_means_4.png", width = 12, height = 12, units = 'in', res = 600)

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
            ylim = c(370, 570), ylab = "ms",
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
#verb
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
            ylim = c(300, 500), ylab = "ms",
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
            ylim = c(0,.5), ylab = "ms",
            xlab = "RC Type", 
            lc = F, uc = F,
            space = 1,
            col = gray(.3),
            main = "regressions, prep phrase")

#matrix verb

bargraph.CI(data = d.matrix, x.factor = rc, 
            response = ffR6, group = length, 
            err.width = 0,
            legend = T, leg.lab = c("long", "short"), 
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

#let's try this with lineplots

png("eyetracking_means_2.png", width = 12, height = 9, units = 'in', res = 600)

par(mfrow = c(3, 4))

lineplot.CI(data = d.NP, x.factor = rc, 
                             response = ff.NP, group = length, 
                             err.width = 0,
                             legend = T, 
                             fixed = T,
                             #leg.lab = c("long", "short"), 
                               x.leg = 1,
                             y.leg = 500,
                             ylim = c(100, 600), ylab = "ms",
                             xlab = "RC Type", 
                             main = "first fix, RC NP")
 
   lineplot.CI(data = d.NP, x.factor = rc, 
                             response = fp.NP, group = length, 
                             err.width = 0,
                             legend = F, 
                             fixed = T,
                             #leg.lab = c("long", "short"), 
                               ylim = c(100, 600), ylab = "ms",
                             xlab = "RC Type", 
                             main = "first pass, RC NP")
 
   lineplot.CI(data = d.NP, x.factor = rc, 
                             response = gp.NP, group = length, 
                             err.width = 0,
                             legend = F, 
                             fixed = T,
                             #leg.lab = c("long", "short"), 
                               ylim = c(100, 600), ylab = "ms",
                             xlab = "RC Type", 
                             main = "go-past, RC NP")
 
   lineplot.CI(data = d.NP, x.factor = rc, 
                             response = ro.NP, group = length, 
                             err.width = 0,
                             legend = F, 
                             fixed = T,
                             #leg.lab = c("long", "short"), 
                               err.lty = 0,
                             ylim = c(0, 1), ylab = "proportion",
                             xlab = "RC Type", 
                             main = "regressions, RC NP")
 
   lineplot.CI(data = d.verb, x.factor = rc, 
                             response = ff.verb, group = length, 
                             err.width = 0,
                             x.leg = 1,
                             y.leg = 350,
                             fixed = T,
                             legend = T, leg.lab = c("long", "short"), 
                             ylim = c(100, 400), ylab = "ms",
                             xlab = "RC Type", 
                             main = "first fix, RC verb")
 
   lineplot.CI(data = d.verb, x.factor = rc, 
                             response = fp.verb, group = length, 
                             err.width = 0,
                             legend = F, 
                             fixed = T,
                             #leg.lab = c("long", "short"), 
                               ylim = c(100, 400), ylab = "ms",
                             xlab = "RC Type", 
                             main = "first pass, RC verb")
 
   lineplot.CI(data = d.verb, x.factor = rc, 
                             response = gp.verb, group = length, 
                             err.width = 0,
                             legend = F, 
                             fixed = T,
                             #leg.lab = c("long", "short"), 
                               ylim = c(100, 400), ylab = "ms",
                             xlab = "RC Type", 
                             main = "go-past, RC verb")
 
   lineplot.CI(data = d.verb, x.factor = rc, 
                             response = ro.verb, group = length, 
                             err.width = 0,
                             legend = F, 
                             fixed = T,
                             #leg.lab = c("long", "short"), 
                               err.lty = 0,
                             ylim = c(0, 1), ylab = "proportion",
                             xlab = "RC Type", 
                             main = "regressions, RC verb")
 
   lineplot.CI(data = d.matrix, x.factor = rc, 
                             response = ffR6, group = length, 
                             err.width = 0,
                             x.leg = 1,
                             y.leg = 400,
                             legend = T, 
                             fixed = T,
                             #leg.lab = c("long", "short"), 
                               ylim = c(100, 450), ylab = "ms",
                             xlab = "RC Type", 
                             main = "first fix, matrix verb")
 
   lineplot.CI(data = d.matrix, x.factor = rc, 
                             response = fpR6, group = length, 
                             err.width = 0,
                             legend = F, 
                             fixed = T,
                             #leg.lab = c("long", "short"), 
                               ylim = c(100, 450), ylab = "ms",
                             xlab = "RC Type", 
                             main = "first pass, matrix verb")
 
   lineplot.CI(data = d.matrix, x.factor = rc, 
                             response = gpR6, group = length, 
                             err.width = 0,
                             legend = F, 
                             fixed = T,
                             #leg.lab = c("long", "short"), 
                               ylim = c(100, 450), ylab = "ms",
                             xlab = "RC Type", 
                             main = "go-past, matrix verb")
 
   lineplot.CI(data = d.matrix, x.factor = rc, 
                             response = roR6, group = length, 
                             err.width = 0,
                             legend = F, 
                             fixed = T,
                             #leg.lab = c("long", "short"), 
                               err.lty = 0,
                             ylim = c(0, 1), ylab = "proportion",
                             xlab = "RC Type", 
                             main = "regressions, matrix verb")
 
dev.off()


#let's make an actual model

#for rc NP
model.ff.NP = lmer(ff.NP~rc.cent * length.cent + (1+rc.cent*length.cent|subj) +
                (1+rc.cent*length.cent|item), data = d.NP)

summary(model.ff.NP)

model.fp.NP = lmer(fp.NP~rc.cent * length.cent + (1+rc.cent*length.cent|subj) +
                     (1+rc.cent*length.cent|item), data = d.NP)

summary(model.fp.NP)

model.gp.NP = lmer(gp.NP~rc.cent * length.cent + (1+rc.cent*length.cent|subj) +
                     (1+rc.cent*length.cent|item), data = d.NP)

summary(model.gp.NP)

model.ro.NP = glmer(ro.NP~rc.cent * length.cent + (1+rc.cent*length.cent|subj) +
                     (1+rc.cent*length.cent|item), data = d.NP, family = "binomial")

summary(model.ro.NP)

#for rc verb

model.ff.verb = lmer(ff.verb~rc.cent * length.cent + (1+rc.cent*length.cent|subj) +
                     (1+rc.cent*length.cent|item), data = d.verb)

summary(model.ff.verb)

model.fp.verb = lmer(fp.verb~rc.cent * length.cent + (1+rc.cent*length.cent|subj) +
                       (1+rc.cent*length.cent|item), data = d.verb)

summary(model.fp.verb)

model.gp.verb = lmer(gp.verb~rc.cent * length.cent + (1+rc.cent*length.cent|subj) +
                       (1+rc.cent*length.cent|item), data = d.verb)

summary(model.gp.verb)

#the full model failed to converge for regressions.  remove 
#slope interactions
model.ro.verb = glmer(ro.verb~rc.cent * length.cent + (1+rc.cent|subj) +
                      (1+rc.cent|item), data = d.verb, family = "binomial")

summary(model.ro.verb)



#prep phrase

model.ff.inter = lmer(ffR5~rc.cent + (1+rc.cent|subj) +
                       (1+rc.cent|item), data = d.inter)

summary(model.ff.inter)

model.fp.inter = lmer(fpR5~rc.cent + (1+rc.cent|subj) +
                        (1+rc.cent|item), data = d.inter)

summary(model.fp.inter)

model.gp.inter = lmer(gpR5~rc.cent + (1+rc.cent|subj) +
                        (1+rc.cent|item), data = d.inter)

summary(model.gp.inter)

model.ro.inter = glmer(roR5~rc.cent + (1+rc.cent|subj) +
                        (1+rc.cent|item), data = d.inter, family = "binomial")

summary(model.ro.inter)

#matrix verb

model.ff.matrix = lmer(ffR6~rc.cent * length.cent + (1+rc.cent*length.cent|subj) +
                       (1+rc.cent*length.cent|item), data = d.matrix)

summary(model.ff.matrix)

#nb:  needed to remove random slope interactions
model.fp.matrix = lmer(fpR6~rc.cent * length.cent + (1+rc.cent + length.cent|subj) +
                         (1+rc.cent + length.cent|item), data = d.matrix)

summary(model.fp.matrix)

model.gp.matrix = lmer(gpR6~rc.cent * length.cent + (1+rc.cent * length.cent|subj) +
                         (1+rc.cent * length.cent|item), data = d.matrix)

summary(model.gp.matrix)

model.ro.matrix = glmer(roR6~rc.cent * length.cent + (1+rc.cent * length.cent|subj) +
                         (1+rc.cent * length.cent|item), data = d.matrix, family = "binomial")

summary(model.ro.matrix)

#let's make another first fix model, to compare each of the other
#conditions to ORC short

d.matrix$cond = as.factor(d.matrix$cond)

model.ff.matrix.new = lmer(ffR6~cond + (1+cond|subj) +
                         (1+cond|item), data = d.matrix)

summary(model.ff.matrix.new)

#and same for first pass

model.fp.matrix.new = lmer(fpR6~cond + (1+cond|subj) +
                             (1+cond|item), data = d.matrix)

summary(model.fp.matrix.new)

#and for go-past

model.gp.matrix.new = lmer(gpR6~cond + (1+cond|subj) +
                             (1+cond|item), data = d.matrix)

summary(model.gp.matrix.new)


#to test pairwise comparisons using multcomp
#I didn't end up using this..

library(multcomp)

#first recompute the model, recoding the two fixed effects
#as a single factor with four levels

d.matrix$rc.length = interaction(d.matrix$rc, d.matrix$length)
model.ff.matrix.2 = lmer(ffR6~rc.length + (1+rc.length|subj) +
                           (1+rc.length|item), data = d.matrix)

summary(model.ff.matrix.2)
#convergence warnings, but does deliver sensible output.

#now feed to glht function

comp = glht(model.ff.matrix.2, linfct = mcp('rc.length' = "Tukey"))
summary(comp)

#and for first pass
#same convergence issues

model.fp.matrix.2 = lmer(fpR6~rc.length + (1+rc.length|subj) +
                           (1+rc.length|item), data = d.matrix)

summary(model.fp.matrix.2)

comp2 = glht(model.fp.matrix.2, linfct = mcp('rc.length' = "Tukey"))
summary(comp2)

#
#let's rename d.matrix, which I'll need for comparison to other exp
d.matrix.2 = d.matrix

#oops, should be called d.matrix.4

d.matrix.4 = d.matrix.2


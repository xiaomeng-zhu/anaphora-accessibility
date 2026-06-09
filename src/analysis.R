library(tidyverse)
library(ggplot2)
library(lme4)
library(bootstrap)
source("helpers.R")

fill_cols = c(
  "#0f9200", # babbage
  "#4A90E2", # davinci
  "#FEE12B", # llama321
  "#F5A623", # llama323
  "#FF2800", # llama318
  "#420D01", # llama318I
  "#000000" # human
)

col_scale = scale_fill_manual(values=fill_cols)

LEVELS = c("babbage",
           "davinci",
           "Llama3-2-1B",
           "Llama3-2-3B",
           "Llama3-1-8B",
           "Llama3-1-8B-Instruct",
           "human")

# ======================== Defining Functions ====================

read_model_res <- function(result_path, model_name) {
  d <- read_csv(result_path)
  d$model <- model_name
  
  return(d)
}
# ======================== Human ====================
human.cleaned <- read_csv("human/human_full_cleaned.csv")
human.cleaned2 <- read_csv("human/human_2nd_cleaned.csv")
human.cleaned3 <- read_csv("human/human_3nd_cleaned.csv")
human <- rbind(human.cleaned, human.cleaned2, human.cleaned3) %>%
  mutate(comp_type=ifelse(grepl(". It is", `Spreadsheet: sentence1`) & (comp_type=="exi>if" | comp_type == "exi>whenever"),
                          paste0(comp_type, "it"), 
                          comp_type)) %>%
  mutate(comp_type=case_when(comp_type=="exi>ifit"~"exi>if.it",
                             comp_type=="exi>if"~"exi>if.he",
                             comp_type=="exi>wheneverit"~"exi>whenever.it",
                             comp_type=="exi>whenever"~"exi>whenever.he",
                             TRUE~comp_type)) %>%
  mutate(comp_type = factor(comp_type, 
                            levels=c("filler",
                                     "exi>uni",
                                     "exi>if.it",
                                     "exi>if.he",
                                     "exi>whenever.it",
                                     "exi>whenever.he",
                                     "existential>verbneg",
                                     "existential_infct>verbneg_infct",
                                     "DNE>verbneg_just",
                                     "DNE_infct>verbneg_just_infct",
                                     "eitheror>conjunction",
                                     "eitheror>eitherposor",
                                     "or>conjunction",
                                     "or>eitherposor")))

fillers <- human %>% filter(comp_type=="filler") %>% group_by(`Participant Private ID`) %>%
  summarize(filler_accuracy = mean(response_correct),
            filler_time = sum(`Reaction Time`)/60000)

good_people_id <- (fillers%>%
                     filter(filler_accuracy >= 0.90))$`Participant Private ID`

tests_filtered <- human %>% 
  filter(comp_type!="filler") %>% 
  filter(`Participant Private ID` %in% good_people_id) %>%
  mutate(model="human") %>%
  rename(comp=comp_type) %>%
  mutate(key=paste(comp, `Spreadsheet: sentence1`, sep="-"))

# write.csv(tests_filtered, "all_human_results.csv", row.names = FALSE)

human_stimuli_list_with_frame <- read_csv("all_human_stimuli.csv") %>%
  mutate(comp_type=case_when(
    (comp_type=="exi>if"&cont=="it")~"exi>if.it",
    (comp_type=="exi>if"&cont=="he")~"exi>if.he",
    (comp_type=="exi>whenever"&cont=="it")~"exi>whenever.it",
    (comp_type=="exi>whenever"&cont=="he")~"exi>whenever.he",
    TRUE~comp_type
  )) %>%
  mutate(key=paste(comp_type, sentence1, sep="-")) %>%
  mutate(subjsubject=paste(subj, subject, sep="-"))

human_tests_filtered <- merge(
  tests_filtered,
  human_stimuli_list_with_frame,
  by="key"
)
write.csv(human_tests_filtered, "human_response.csv", row.names=FALSE)



# ============================ Experiment 1A ============================ 

a.every.babbage <- read_model_res("results/accuracy/exp1a_babbage.csv", "babbage")
a.every.davinci <- read_model_res("results/accuracy/exp1a_davinci.csv", "davinci")
a.every.Llama318BI <- read_model_res("results/accuracy/exp1a_Llama3-1-8B-Instruct.csv", "Llama3-1-8B-Instruct")
a.every.Llama318B <- read_model_res("results/accuracy/exp1a_Llama3-1-8B.csv", "Llama3-1-8B")
a.every.Llama321B <- read_model_res("results/accuracy/exp1a_Llama3-2-1B.csv", "Llama3-2-1B")
a.every.Llama323B <- read_model_res("results/accuracy/exp1a_Llama3-2-3B.csv", "Llama3-2-3B")

## summarizing human and model results
a.every.model.summarized <- rbind(a.every.babbage,
                          a.every.davinci,
                          a.every.Llama318BI,
                          a.every.Llama318B,
                          a.every.Llama321B,
                          a.every.Llama323B) %>%
  mutate(gender = case_when(gender=="f"~"She",
                            gender=="m"~"He",
                            gender=="p"~"They")) %>%
  filter(comp == "every_diff>a_diff" & gender != "They") %>%
  mutate(comp = case_when(comp=="every_diff>a_diff"~"exi>every")) %>%
  group_by(comp, model) %>%
  summarize(accuracy=mean(correct), ci_low = ci.low(correct), ci_high=ci.high(correct))
a.every.human.summarized <- tests_filtered %>% 
  filter(comp == "exi>uni") %>%
  mutate(comp = "exi>every") %>%
  group_by(comp, model) %>%
  summarize(accuracy=mean(response_correct), ci_low=ci.low(response_correct), ci_high=ci.high(response_correct))
  
## combining human with model
a.every.summarized <- rbind(a.every.model.summarized, a.every.human.summarized) %>%
  mutate(model = factor(model,
                        levels=LEVELS))

## plot human with model
a.every.plot <- a.every.summarized %>%
  ggplot(aes(x=comp, fill=model, y=accuracy)) +
  geom_bar(stat="identity", alpha=0.8, position=position_dodge()) +
  theme(axis.text.x = element_text(vjust = 0.7),
        legend.position = "bottom",
        legend.key.size = unit(0.3, 'cm')) +
  guides(fill = guide_legend(nrow = 1, byrow=TRUE)) +
  xlab("Comparison Type") +
  ylab("% Expected") + 
  geom_hline(yintercept = 0.5, color="#666666", lty=2) + 
  geom_errorbar(aes(ymin=accuracy - ci_low, ymax=accuracy + ci_high),
                width = 0.4,
                col="#666666",
                position=position_dodge(width=0.9)) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.3, 'cm')) + guides(fill = guide_legend(nrow = 2, byrow=TRUE)) + 
  col_scale 


# =============== Experiment 1B ================
exp1b.babbage.name <- read_model_res("results/accuracy/exp1b_babbage_name.csv", "babbage")
exp1b.davinci.name <- read_model_res("results/accuracy/exp1b_davinci_name.csv", "davinci")
exp1b.Llama318B.name <- read_model_res("results/accuracy/exp1b_Llama3-1-8B_name.csv", "Llama3-1-8B")
exp1b.Llama318BI.name <- read_model_res("results/accuracy/exp1b_Llama3-1-8B-Instruct_name.csv", "Llama3-1-8B-Instruct")
exp1b.Llama321B.name <- read_model_res("results/accuracy/exp1b_Llama3-2-1B_name.csv", "Llama3-2-1B")
exp1b.Llama323B.name <- read_model_res("results/accuracy/exp1b_Llama3-2-3B_name.csv", "Llama3-2-3B")

exp1b.model.summarized <- rbind(exp1b.babbage.name, exp1b.davinci.name, exp1b.Llama318B.name, exp1b.Llama318BI.name, exp1b.Llama321B.name, exp1b.Llama323B.name) %>% 
  mutate(model = factor(model, levels=c("babbage",
                                        "davinci",
                                        "Llama3-2-1B",
                                        "Llama3-2-3B",
                                        "Llama3-1-8B",
                                        "Llama3-1-8B-Instruct"))) %>%
  mutate(comp = case_when(comp=="if_diff>a_diff"~"exi>if",
                          comp=="whenever_diff>a_diff"~"exi>whenever")) %>%
  group_by(comp, model, cont) %>%
  summarize(accuracy=mean(correct), ci_low = ci.low(correct), ci_high=ci.high(correct))

exp1b.human.summarized <- tests_filtered %>% 
  filter(comp %in% c("exi>if.he", "exi>if.it", "exi>whenever.it", "exi>whenever.he")) %>%
  mutate(cont = case_when(comp == "exi>if.it"~"it",
                          comp == "exi>if.he"~"he",
                          comp == "exi>whenever.it"~"it",
                          comp == "exi>whenever.he"~"he")) %>%
  mutate(comp = case_when(comp == "exi>if.it"~"exi>if",
                               comp == "exi>if.he"~"exi>if",
                               comp == "exi>whenever.it"~"exi>whenever",
                               comp == "exi>whenever.he"~"exi>whenever")) %>%
  group_by(comp, model, cont) %>%
  summarize(accuracy=mean(response_correct), ci_low=ci.low(response_correct), ci_high=ci.high(response_correct))
  
exp1b.summarized <- rbind(exp1b.model.summarized, exp1b.human.summarized) %>%
  mutate(model = factor(model, levels=LEVELS))
exp1b.summarized.it <- exp1b.summarized %>% 
  filter(cont == "it") %>%
  select(comp, model, accuracy, ci_low, ci_high)
exp1b.summarized.he <- exp1b.summarized %>% 
  filter(cont == "he") %>%
  select(comp, model, accuracy, ci_low, ci_high)

exp1.summarized <- rbind(a.every.summarized, exp1b.summarized.it) %>%
  mutate(comp = factor(comp, levels=c("exi>every", "exi>if", "exi>whenever")))

exp1.plot <- exp1.summarized %>%
  ggplot(aes(x=comp, fill=model, y=accuracy)) +
  geom_bar(stat="identity", alpha=0.8, position=position_dodge()) +
  theme(axis.text.x = element_text(vjust = 0.7),
        legend.position = "bottom",
        legend.key.size = unit(0.3, 'cm')) +
  guides(fill = guide_legend(nrow = 1, byrow=TRUE)) +
  xlab("Comparison Type") +
  ylab("% Expected") + 
  geom_hline(yintercept = 0.5, color="#666666", lty=2) + 
  geom_errorbar(aes(ymin=accuracy - ci_low, ymax=accuracy + ci_high),
                width = 0.4,
                col="#666666",
                position=position_dodge(width=0.9)) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.3, 'cm')) + guides(fill = guide_legend(nrow = 1, byrow=TRUE)) + col_scale
ggsave("plots/exp1.pdf", exp1.plot, width=8, height=1.5, dpi=300)

exp1.he <- exp1b.summarized.he %>%
  ggplot(aes(x=comp, fill=model, y=accuracy)) +
  geom_bar(stat="identity", alpha=0.8, position=position_dodge()) +
  theme(axis.text.x = element_text(vjust = 0.7),
        legend.position = "bottom",
        legend.key.size = unit(0.3, 'cm')) +
  guides(fill = guide_legend(nrow = 1, byrow=TRUE)) +
  xlab("Comparison Type") +
  ylab("% Expected") + 
  geom_hline(yintercept = 0.5, color="#666666", lty=2) + 
  geom_errorbar(aes(ymin=accuracy - ci_low, ymax=accuracy + ci_high),
                width = 0.4,
                col="#666666",
                position=position_dodge(width=0.9)) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.3, 'cm')) + guides(fill = guide_legend(nrow = 2, byrow=TRUE)) + col_scale
ggsave("plots/exp1he.pdf", exp1.he, width=5, height=2, dpi=300)
# =================== Experiment 2 =======================

neg.babbage <- read_model_res("results/accuracy/exp2_babbage.csv", "babbage")
neg.davinci <- read_model_res("results/accuracy/exp2_davinci.csv", "davinci")
neg.Llama318B <- read_model_res("results/accuracy/exp2_Llama3-1-8B.csv", "Llama3-1-8B")
neg.Llama318BI <- read_model_res("results/accuracy/exp2_Llama3-1-8B-Instruct.csv", "Llama3-1-8B-Instruct")
neg.Llama321B <- read_model_res("results/accuracy/exp2_Llama3-2-1B.csv", "Llama3-2-1B")
neg.Llama323B <- read_model_res("results/accuracy/exp2_Llama3-2-3B.csv", "Llama3-2-3B")

neg.model.summarized <- rbind(neg.babbage, neg.davinci, neg.Llama318B, neg.Llama318BI, neg.Llama321B, neg.Llama323B) %>%
  filter(comp %in% c("DNE>negation2_just", 
                     "existential>negation2", 
                     "DNE_infact>negation2_just_infact", 
                     "existential_infact>negation2_infact")) %>%
  mutate(comp = case_when(comp=="DNE>negation2_just"~"DN>Neg",
                          comp=="existential>negation2"~"Exi>Neg",
                          comp=="DNE_infact>negation2_just_infact"~"DN>Neg(infact)",
                          comp=="existential_infact>negation2_infact"~"Exi>Neg(infact)")) %>%
  group_by(comp, model) %>% 
  summarize(accuracy=mean(correct), ci_low = ci.low(correct), ci_high = ci.high(correct))

neg.human.summarized <- tests_filtered %>%
  filter(comp %in% c("DNE>verbneg_just",
                     "DNE_infct>verbneg_just_infct",
                     "existential>verbneg",
                     "existential_infct>verbneg_infct")) %>%
  mutate(comp = case_when(comp=="DNE>verbneg_just"~"DN>Neg",
                          comp=="existential>verbneg"~"Exi>Neg",
                          comp=="DNE_infct>verbneg_just_infct"~"DN>Neg(infact)",
                          comp=="existential_infct>verbneg_infct"~"Exi>Neg(infact)")) %>%
  group_by(comp, model) %>%
  summarize(accuracy=mean(response_correct), ci_low = ci.low(response_correct), ci_high = ci.high(response_correct))

neg.summarizd <- rbind(neg.model.summarized, neg.human.summarized) %>%
  mutate(model=factor(model, levels=LEVELS),
         comp=factor(comp, levels=c("Exi>Neg",
                                    "DN>Neg",
                                    "Exi>Neg(infact)",
                                    "DN>Neg(infact)")))


neg.accuracy <- neg.summarizd %>%
  ggplot(aes(x=model, fill=model, y=accuracy)) +
  geom_bar(stat="identity", alpha=0.8, position=position_dodge()) +
  geom_hline(yintercept = 0.5, color="#666666", lty=2) + 
  theme(axis.text.x = element_blank(),
        legend.position = "bottom",
        legend.key.size = unit(0.3, 'cm')) + 
  guides(fill = guide_legend(nrow = 3, byrow=TRUE)) +
  ylab("% Expected") +
  facet_wrap(~comp, ncol=2)+
  geom_errorbar(aes(ymin=accuracy - ci_low, ymax=accuracy + ci_high),
                width = 0.4,
                col="#666666",
                position=position_dodge(width=0.9)) + col_scale

ggsave("plots/exp2.pdf", neg.accuracy, width=5, height=3, dpi=300)

# =================== Experiment 3 ====================

andor.babbage <- read_model_res("results/accuracy/exp3_babbage.csv", "babbage")
andor.davinci <- read_model_res("results/accuracy/exp3_davinci.csv", "davinci")
andor.Llama318B <- read_model_res("results/accuracy/exp3_Llama3-1-8B.csv", "Llama3-1-8B")
andor.Llama318BI <- read_model_res("results/accuracy/exp3_Llama3-1-8B-Instruct.csv", "Llama3-1-8B-Instruct")
andor.Llama321B <- read_model_res("results/accuracy/exp3_Llama3-2-1B.csv", "Llama3-2-1B")
andor.Llama323B <- read_model_res("results/accuracy/exp3_Llama3-2-3B.csv", "Llama3-2-3B")

andor.model.summarized <- rbind(andor.babbage, andor.davinci, andor.Llama318B, andor.Llama318BI, andor.Llama321B, andor.Llama323B) %>%
  group_by(comp, model) %>%
  filter(comp %in% c("disjunction_eitheror>conjunction", "disjunction_or>conjunction", "disjunction_eitheror>disjunction_eitheror_pos", "disjunction_or>disjunction_eitheror_pos")) %>%
  mutate(comp=case_when(comp=="disjunction_eitheror>conjunction"~"eitheror>conjunction",
                        comp=="disjunction_or>conjunction"~"or>conjunction",
                        comp=="disjunction_eitheror>disjunction_eitheror_pos"~"eitheror>either(pos)or",
                        comp=="disjunction_or>disjunction_eitheror_pos"~"or>either(pos)or")) %>%
  summarize(accuracy=mean(correct), ci_low = ci.low(correct), ci_high = ci.high(correct))

andor.human.summarized <- tests_filtered %>%
  filter(comp %in% c("eitheror>eitherposor",
                     "eitheror>conjunction",
                     "or>eitherposor",
                     "or>conjunction")) %>%
  mutate(comp = case_when(comp=="eitheror>eitherposor"~"eitheror>either(pos)or",
                          comp=="or>eitherposor"~"or>either(pos)or",
                          TRUE~comp)) %>%
  group_by(comp, model) %>%
  summarize(accuracy=mean(response_correct), ci_low = ci.low(response_correct), ci_high = ci.high(response_correct))
  
andor.summarized <- rbind(andor.model.summarized, andor.human.summarized) %>%
  mutate(model = factor(model, levels=LEVELS))
  
andor.accuracy <- andor.summarized %>% 
  ggplot(aes(x=comp, fill=model, y=accuracy)) +
  geom_bar(stat="identity", alpha=0.8, position=position_dodge()) +
  geom_hline(yintercept = 0.5, color="#666666", lty=2) + 
  theme(axis.text.x = element_text(vjust = 0.7),
        legend.position = "bottom",
        legend.key.size = unit(0.3, 'cm')) + 
  guides(fill = guide_legend(nrow = 1, byrow=TRUE)) +
  ylim(0,1)+
  xlab("Comparison Type") + 
  ylab("% Expected") +
  geom_errorbar(aes(ymin=accuracy - ci_low, ymax=accuracy + ci_high),
                width = 0.4,
                col="#666666",
                position=position_dodge(width=0.9)) + col_scale

ggsave("plots/exp3.pdf", andor.accuracy, height=1.5, width=8, dpi=300)

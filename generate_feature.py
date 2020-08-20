from preprocess import Featurizer
import os
import numpy as np
import pandas as pd
import datetime


def read_txt(path, corpus):
	with open(path, encoding="utf-8") as f:
		for line in f.readlines():
			line = line.rstrip("\n")
			score = line.split(",")[0]
			text = ",".join(line.split(",")[1:])
			corpus[text] = score

def get_data(cutoff):
	corpus = {}
	i = 0
	for root, dir, files in os.walk("D:\\work\\str_out_score"):
		for file in files:
			file_path = os.path.join(root, file)
			read_txt(file_path, corpus)
			i += 1
			if i >= cutoff:
				break
	return corpus


class Preprocessing():
	def __init__(self, string, score):
		f = Featurizer()
		self.feature = {}
		self.feature["string_length"] = f.string_length(string)
		self.feature["has_english_text"] = f.has_english_text(string)
		self.feature["entropy_rate"] = f.entropy_rate(string)
		self.feature["english_letter_freq_div"] = f.english_letter_freq_div(string)
		self.feature["average_scrabble_score"] = f.average_scrabble_score(string)
		self.feature["whitespace_percentage"] = f.whitespace_percentage(string)
		self.feature["alpha_percentage"] = f.alpha_percentage(string)
		self.feature["digit_percentage"] = f.digit_percentage(string)
		self.feature["punctuation_percentage"] = f.punctuation_percentage(string)
		self.feature["vowel_consenant_ratio"] = f.vowel_consenant_ratio(string)
		self.feature["capital_letter_ratio"] = f.capital_letter_ratio(string)
		self.feature["title_words_ratio"] = f.title_words_ratio(string)
		self.feature["average_word_length"] = f.average_word_length(string)
		self.feature["has_ip"] = f.has_ip(string)
		self.feature["has_ip_srv"] = f.has_ip_srv(string)
		self.feature["has_url"] = f.has_url(string)
		self.feature["has_email"] = f.has_email(string)
		self.feature["has_fqdn"] = f.has_fqdn(string)
		self.feature["has_namespace"] = f.has_namespace(string)
		self.feature["has_msword_version"] = f.has_msword_version(string)
		self.feature["has_packer"] = f.has_packer(string)
		self.feature["has_crypto_related"] = f.has_crypto_related(string)
		self.feature["is_blacklisted"] = f.is_blacklisted(string)
		self.feature["has_privilege_constant"] = f.has_privilege_constant(string)
		self.feature["has_mozilla_api"] = f.has_mozilla_api(string)
		self.feature["is_strict_fqdn"] = f.is_strict_fqdn(string)
		self.feature["has_hive_name"] = f.has_hive_name(string)
		self.feature["is_mac"] = f.is_mac(string)
		self.feature["has_extension"] = f.has_extension(string)
		self.feature["is_md5"] = f.is_md5(string)
		self.feature["is_sha1"] = f.is_sha1(string)
		self.feature["is_sha256"] = f.is_sha256(string)
		self.feature["is_irrelevant_windows_api"] = f.is_irrelevant_windows_api(string)
		self.feature["has_guid"] = f.has_guid(string)
		self.feature["is_antivirus"] = f.is_antivirus(string)
		self.feature["is_whitelisted"] = f.is_whitelisted(string)
		self.feature["is_common_dll"] = f.is_common_dll(string)
		self.feature["is_boost_lib"] = f.is_boost_lib(string)
		self.feature["is_delphi_lib"] = f.is_delphi_lib(string)
		self.feature["has_event"] = f.has_event(string)
		self.feature["is_registry"] = f.is_registry(string)
		self.feature["has_malware_identifier"] = f.has_malware_identifier(string)
		self.feature["has_sid"] = f.has_sid(string)
		self.feature["has_keylogger"] = f.has_keylogger(string)
		self.feature["has_oid"] = f.has_oid(string)
		self.feature["has_product_id"] = f.has_product_id(string)
		self.feature["is_oss"] = f.is_oss(string)
		self.feature["is_user_agent"] = f.is_user_agent(string)
		self.feature["has_sddl"] = f.has_sddl(string)
		self.feature["has_protocol"] = f.has_protocol(string)
		self.feature["is_protocol_method"] = f.is_protocol_method(string)
		self.feature["is_base64"] = f.is_base64(string)
		self.feature["is_hex_not_numeric_not_alpha"] = f.is_hex_not_numeric_not_alpha(string)
		self.feature["has_format_specifier"] = f.has_format_specifier(string)
		self.feature["ends_with_line_feed"] = f.ends_with_line_feed(string)
		self.feature["has_path"] = f.has_path(string)
		self.feature["has_pdb"] = f.has_pdb(string)
		self.feature["has_privilege"] = f.has_privilege(string)
		self.feature["is_known_xml"] = f.is_known_xml(string)
		self.feature["is_cpp_runtime"] = f.is_cpp_runtime(string)
		self.feature["is_library"] = f.is_library(string)
		self.feature["is_date"] = f.is_date(string)
		self.feature["is_pe_artifact"] = f.is_pe_artifact(string)
		self.feature["has_public_key"] = f.has_public_key(string)
		self.feature["markov_junk"] = f.markov_junk(string)
		self.feature["is_x86"] = f.is_x86(string)
		self.feature["is_common_path"] = f.is_common_path(string)
		self.feature["is_code_page"] = f.is_code_page(string)
		self.feature["is_language"] = f.is_language(string)
		self.feature["is_region_tag"] = f.is_region_tag(string)
		self.feature["has_not_latin"] = f.has_not_latin(string)
		self.feature["is_known_folder"] = f.is_known_folder(string)
		self.feature["is_malware_api"] = f.is_malware_api(string)
		self.feature["is_environment_variable"] = f.is_environment_variable(string)
		self.feature["has_variable_name"] = f.has_variable_name(string)
		self.feature["has_padding_string"] = f.has_padding_string(string)
		self.feature["string"] = string
		self.feature["score"] = score


if __name__ == "__main__":
	f = Featurizer()
	all_features = f.features
	corpus = get_data(100)
	print(len(corpus))
	
	begin = datetime.datetime.now()

	content = []
	for text, score in corpus.items():
		p = Preprocessing(text, score)
		feature = p.feature
		content.append(list(feature.values()))

	end = datetime.datetime.now()
	print("time consumed:" + str((end - begin).seconds) + "seconds")

	df = pd.DataFrame(content)
	df = df.rename(columns={0: "string_length",
							1: "has_english_text",
							2: "entropy_rate",
							3: "english_letter_freq_div",
							4: "average_scrabble_score",
							5: "whitespace_percentage",
							6: "alpha_percentage",
							7: "digit_percentage",
							8: "punctuation_percentage",
							9: "vowel_consenant_ratio",
							10: "capital_letter_ratio",
							11: "title_words_ratio",
							12: "average_word_length",
							13: "has_ip",
							14: "has_ip_srv",
							15: "has_url",
							16: "has_email",
							17: "has_fqdn",
							18: "has_namespace",
							19: "has_msword_version",
							20: "has_packer",
							21: "has_crypto_related",
							22: "is_blacklisted",
							23: "has_privilege_constant",
							24: "has_mozilla_api",
							25: "is_strict_fqdn",
							26: "has_hive_name",
							27: "is_mac",
							28: "has_extension",
							29: "is_md5",
							30: "is_sha1",
							31: "is_sha256",
							32: "is_irrelevant_windows_api",
							33: "has_guid",
							34: "is_antivirus",
							35: "is_whitelisted",
							36: "is_common_dll",
							37: "is_boost_lib",
							38: "is_delphi_lib",
							39: "has_event",
							40: "is_registry",
							41: "has_malware_identifier",
							42: "has_sid",
							43: "has_keylogger",
							44: "has_oid",
							45: "has_product_id",
							46: "is_oss",
							47: "is_user_agent",
							48: "has_sddl",
							49: "has_protocol",
							50: "is_protocol_method",
							51: "is_base64",
							52: "is_hex_not_numeric_not_alpha",
							53: "has_format_specifier",
							54: "ends_with_line_feed",
							55: "has_path",
							56: "has_pdb",
							57: "has_privilege",
							58: "is_known_xml",
							59: "is_cpp_runtime",
							60: "is_library",
							61: "is_date",
							62: "is_pe_artifact",
							63: "has_public_key",
							64: "markov_junk",
							65: "is_x86",
							66: "is_common_path",
							67: "is_code_page",
							68: "is_language",
							69: "is_region_tag",
							70: "has_not_latin",
							71: "is_known_folder",
							72: "is_malware_api",
							73: "is_environment_variable",
							74: "has_variable_name",
							75: "has_padding_string",
							76: "string",
							77: "score"})
	df.to_csv("77_features.csv")


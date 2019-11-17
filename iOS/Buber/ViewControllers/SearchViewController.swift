//
//  SearchViewController.swift
//  Buber
//
//  Created by Alexey Salangin on 11/16/19.
//  Copyright © 2019 Alexey Salangin. All rights reserved.
//

import UIKit
import SnapKit
import Keyboardy

protocol SearchViewControllerDelegate: AnyObject {
    func searchViewController(vc: SearchViewController, searchText: String)
}

final class SearchViewController: UIViewController {
    weak var delegate: SearchViewControllerDelegate?

    private let whereToSearchField = UITextField()
    private let searchButton = UIButton()
    private let searchImageView: UIImageView
    private var keyboardHeight: CGFloat = 0

    init() {
        self.searchImageView = Self.makeSearchImageView()
        super.init(nibName: nil, bundle: nil)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        self.view.backgroundColor = .white

        self.setupSubviews()
    }

    func showKeyboard() {
        self.whereToSearchField.becomeFirstResponder()
    }

    func clearTextField() {
        self.whereToSearchField.text = nil
    }

    private func setupSubviews() {
        self.view.addSubview(self.searchImageView)
        self.view.addSubview(self.whereToSearchField)
        self.view.addSubview(self.searchButton)

        self.searchImageView.snp.makeConstraints { make in
            make.width.height.equalTo(16)
            make.centerY.equalTo(self.searchButton)
            make.left.equalToSuperview().offset(16)
        }

        self.whereToSearchField.snp.makeConstraints { make in
            make.top.equalToSuperview().offset(32)
            make.left.equalTo(self.searchImageView.snp.right).offset(16)
        }
        self.whereToSearchField.contentVerticalAlignment = .center

        self.searchButton.snp.makeConstraints { make in
            make.top.equalToSuperview().offset(32)
            make.left.equalTo(self.whereToSearchField.snp.right).offset(8)
            make.right.equalToSuperview().offset(-16)
            make.width.equalTo(60)
            make.height.equalTo(self.whereToSearchField.snp.height)
        }

        self.whereToSearchField.placeholder = "Where to?"
        self.whereToSearchField.font = UIFont.appMedium(16)
        self.whereToSearchField.spellCheckingType = .no

        let attributedTitle = NSAttributedString(
            string: "Search",
            attributes: [
                .foregroundColor: UIColor.app,
                .font: UIFont.appMedium(16)
            ]
        )
        self.searchButton.setAttributedTitle(attributedTitle, for: .normal)
        self.searchButton.addTarget(self, action: #selector(self.didTapSearchButton), for: .touchUpInside)
    }

    @objc
    private func didTapSearchButton() {
        self.view.endEditing(true)
        self.delegate?.searchViewController(vc: self, searchText: self.whereToSearchField.text ?? "")
    }

    private static func makeSearchImageView() -> UIImageView {
        let imageView = UIImageView()
        imageView.image = UIImage(named: "search")
        imageView.tintColor = .placeholderText
        return imageView
    }
}
